from cmath import nan
from dis import dis
import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from bitrap.utils.visualization import Visualizer
from bitrap.utils.box_utils import cxcywh_to_x1y1x2y2
from bitrap.utils.dataset_utils import restore
from bitrap.modeling.gmm2d import GMM2D
from bitrap.modeling.gmm4d import GMM4D
from .evaluate import evaluate_multimodal, compute_kde_nll
from .utils import print_info, viz_results, post_process

from tqdm import tqdm
import pickle as pkl
import pdb
import time
import pandas as pd 

import sys
import bitrap.engine.ogl_viewer.viewer as gl
import pyzed.sl as sl
import cv2
import math
import yaml



class person:
    def __init__(self,id,point_box,bbox,dist,height,frame_rate=15): # takes the id and first bounding box of the detected person
        self.id = id
        self.input_size = 8          # input size for point based is 8 frames
        self.index = self.input_size-1
        self.distance = dist
        self.height = height
        self.bounding = bbox
        self.history = np.zeros((self.input_size, 6)) # 6 parameters position(x,y)velocity(x,y),acceleration(x,y) 
        self.dt = 4  #1/frame_rate            # change in time
        self.add_box(point_box)
   
    def derivative(self,x, dt=1):
        not_nan_mask = ~np.isnan(x)
        masked_x = x[not_nan_mask]

        if masked_x.shape[-1] < 2:
            return np.zeros_like(x)
        dx = np.full_like(x, np.nan)
    
        dx[not_nan_mask] = np.ediff1d(masked_x, to_begin=(masked_x[1] - masked_x[0]))/ self.dt
        
        return dx
             

    def add_box(self,box):
        self.history[self.index] = box
        for j in range (self.index,self.input_size-1):
            self.history[j] =  self.history[j+1]
           
        # calculate the acceleration  a1 = V2-v1 / dt --> dt depends on the frame rate 
        self.history[self.input_size-1] =box       # The last is always updated by the new value
        x_acc = self.derivative( self.history[:,2])          # -> position y , box[2] is velocity x
        y_acc = self.derivative( self.history[:,3])          # box[3] is velocity y

        self.history[:,4] =  x_acc 
        self.history[:,5] =  y_acc 
        
        if self.index !=0: 
            self.index-=1
        
   
    


def do_train(cfg, epoch, model, optimizer, dataloader, device, logger=None, lr_scheduler=None):
    model.train()
    max_iters = len(dataloader)
    if cfg.DATASET.NAME in ['eth', 'hotel', 'univ', 'zara1', 'zara2']:
        viz = Visualizer(mode='plot')
    else:
        viz = Visualizer(mode='image')
    with torch.set_grad_enabled(True):
        for iters, batch in enumerate(tqdm(dataloader), start=1):
            X_global = batch['input_x'].to(device)
            y_global = batch['target_y'].to(device)
            img_path = batch['cur_image_file']
            resolution = batch['pred_resolution'].numpy()

            if cfg.DATASET.NAME in ['eth', 'hotel', 'univ', 'zara1', 'zara2']:
                input_x = batch['input_x_st'].to(device)
                neighbors_st = restore(batch['neighbors_x_st'])
                adjacency = restore(batch['neighbors_adjacency'])
                first_history_indices = batch['first_history_index']

            else:
                input_x = X_global
                neighbors_st, adjacency, first_history_indices = None, None, None
            

            pred_goal, pred_traj, loss_dict, dist_goal, dist_traj = model(input_x, 
                                                                    y_global, 
                                                                    neighbors_st=neighbors_st, 
                                                                    adjacency=adjacency, 
                                                                    cur_pos=X_global[:, -1, :cfg.MODEL.DEC_OUTPUT_DIM],
                                                                    first_history_indices=first_history_indices)
            if cfg.MODEL.LATENT_DIST == 'categorical':
                loss = loss_dict['loss_goal'] + \
                       loss_dict['loss_traj'] + \
                       model.param_scheduler.kld_weight * loss_dict['loss_kld'] - \
                       1. * loss_dict['mutual_info_p']
            else:
                loss = loss_dict['loss_goal'] + \
                       loss_dict['loss_traj'] + \
                       model.param_scheduler.kld_weight * loss_dict['loss_kld']
            model.param_scheduler.step()
            loss_dict = {k:v.item() for k, v in loss_dict.items()}
            loss_dict['lr'] = optimizer.param_groups[0]['lr']
            # optimize
            optimizer.zero_grad() # avoid gradient accumulate from loss.backward()
            loss.backward()
            
            # loss_dict['grad_norm'] = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            torch.nn.utils.clip_grad_value_(model.parameters(), 1.0)
            optimizer.step()

            if cfg.SOLVER.scheduler == 'exp':
                lr_scheduler.step()
            if iters % cfg.PRINT_INTERVAL == 0:
                print_info(epoch, model, optimizer, loss_dict, logger)

            if cfg.VISUALIZE and iters % max(int(len(dataloader)/5), 1) == 0:
                ret = post_process(cfg, X_global, y_global, pred_traj, pred_goal=pred_goal, dist_goal=dist_goal)
                X_global, y_global, pred_goal, pred_traj, dist_traj, dist_goal = ret
                viz_results(viz, X_global, y_global, pred_traj, img_path, dist_goal, dist_traj,
                            bbox_type=cfg.DATASET.BBOX_TYPE, normalized=False, logger=logger, name='pred_train')
                
def do_val(cfg, epoch, model, dataloader, device, logger=None):
    model.eval()
    loss_goal_val = 0.0
    loss_traj_val = 0.0
    loss_KLD_val = 0.0
    with torch.set_grad_enabled(False):
        for iters, batch in enumerate(tqdm(dataloader), start=1):
            X_global = batch['input_x'].to(device)
            y_global = batch['target_y'].to(device)
            img_path = batch['cur_image_file']
            # For ETH_UCY dataset only
            if cfg.DATASET.NAME in ['eth', 'hotel', 'univ', 'zara1', 'zara2']:
                input_x = batch['input_x_st'].to(device)
                neighbors_st = restore(batch['neighbors_x_st'])
                adjacency = restore(batch['neighbors_adjacency'])
                first_history_indices = batch['first_history_index']
            else:
                input_x = X_global
                neighbors_st, adjacency, first_history_indices = None, None, None
            
            pred_goal, pred_traj, loss_dict, _, _ = model(input_x, 
                                                            y_global, 
                                                            neighbors_st=neighbors_st,
                                                            adjacency=adjacency,
                                                            cur_pos=X_global[:, -1, :cfg.MODEL.DEC_OUTPUT_DIM],
                                                            first_history_indices=first_history_indices)

            # compute loss
            loss = loss_dict['loss_goal'] + loss_dict['loss_traj'] + loss_dict['loss_kld']
            loss_goal_val += loss_dict['loss_goal'].item()
            loss_traj_val += loss_dict['loss_traj'].item()
            loss_KLD_val += loss_dict['loss_kld'].item()
    loss_goal_val /= (iters + 1)
    loss_traj_val /= (iters + 1)
    loss_KLD_val /= (iters + 1)
    loss_val = loss_goal_val + loss_traj_val + loss_KLD_val
    
    info = "loss_val:{:.4f}, \
            loss_goal_val:{:.4f}, \
            loss_traj_val:{:.4f}, \
            loss_kld_val:{:.4f}".format(loss_val, loss_goal_val, loss_traj_val, loss_KLD_val)
        
    if hasattr(logger, 'log_values'):
        logger.info(info)
        logger.log_values({'loss_val':loss_val, 
                           'loss_goal_val':loss_goal_val,
                           'loss_traj_val':loss_traj_val, 
                           'loss_kld_val':loss_KLD_val})#, step=epoch)
    else:
        print(info)
    return loss_val

#def inference(cfg, epoch, model, dataloader, device, logger=None, eval_kde_nll=False, test_mode=False):
def inference(cfg,cfg_zed, epoch, model, device, logger=None, eval_kde_nll=False, test_mode=False):

    # dataloader = make_dataloader(cfg, 'test')
    model.eval()

    all_img_paths = []
    all_X_globals = []
    all_pred_goals = []
    all_gt_goals = []
    all_pred_trajs = []
    all_gt_trajs = []
    all_distributions = []
    all_timesteps = []
    if cfg.DATASET.NAME in ['eth', 'hotel', 'univ', 'zara1', 'zara2']:
        viz = Visualizer(mode='plot')
    else:
        viz = Visualizer(mode='image')

# Total time taken by all
    all_time = []
    all_total_time = []
    all_average = []
    total_time = 0
    index_batch = 1    # batch number used to find average



#################################################################   ZED CAMERA ###############################################################
    # Create a Camera object
    zed = sl.Camera()
    zed_pose = sl.Pose()

    # Create a InitParameters object and set configuration parameters
    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.HD1080  # Use HD1080 video mode    
    init_params.coordinate_units = sl.UNIT.METER
    init_params.camera_fps = cfg_zed['frame_rate']                          # Set fps at 30
    init_params.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP
    #camera_position = zed.get_position(zed_pose, sl.REFERENCE_FRAME.CAMERA)
    
    # If applicable, use the SVO given as parameter
    # Otherwise use ZED live stream
    if len(sys.argv) == 2:
        filepath = sys.argv[1]
        print("Using SVO file: {0}".format(filepath))
        init_params.set_from_svo_file(filepath)

    # Set runtime parameters
    runtime_parameters = sl.RuntimeParameters()

    # Open the camera
    err = zed.open(init_params)
    if err != sl.ERROR_CODE.SUCCESS:
        exit(1)

    # Enable object detection module
    obj_param = sl.ObjectDetectionParameters()
    # Defines if the object detection will track objects across images flow.
    obj_param.enable_tracking = True       # if True, enable positional tracking

    if obj_param.enable_tracking:
        zed.enable_positional_tracking()
        
    zed.enable_object_detection(obj_param)

    camera_info = zed.get_camera_information()
    # Create OpenGL viewer
    # viewer = gl.GLViewer()
    # viewer.init(camera_info.calibration_parameters.left_cam, obj_param.enable_tracking)

    # Configure object detection runtime parameters
    obj_runtime_param = sl.ObjectDetectionRuntimeParameters()
    obj_runtime_param.detection_confidence_threshold = 60
    obj_runtime_param.object_class_filter = [sl.OBJECT_CLASS.PERSON]    # Only detect Persons

    # Create ZED objects filled in the main loop
    objects = sl.Objects()
    image = sl.Mat()
    # image_zed = sl.Mat(image_size.width, image_size.height, sl.MAT_TYPE.U8_C4)
    # depth_image_zed = sl.Mat(image_size.width, image_size.height, sl.MAT_TYPE.U8_C4)
    
    objects_dic= dict()          # Dictionary holding objects values        
    
    while zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:#viewer.is_available():
        # Grab an image, a RuntimeParameters object must be given to grab()
        # if zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
        # Retrieve left image
        zed.retrieve_image(image, sl.VIEW.LEFT)
        

        # Retrieve objects
        zed.retrieve_objects(objects, obj_runtime_param)
        # Update GL view
        #viewer.update_view(image, objects)
        

        batch_dic=dict()           # new batch dictionary
        prediction_input=[]          # List holding input for prediction
        prediction_Xglobal=[] 
        obj_detected = False # False
        prediction_input_dist =[]
        prediction_input_height =[]
        bbox_list = []
        if objects.is_new:
            # Count the number of objects detected
            #print("{} Object(s) detected".format(len(objects.object_list)))

            if len(objects.object_list):
                # # Display the 3D location of an object
                # first_object = objects.object_list[0]
                # position = first_object.position
                # # print(" 3D position : [{0},{1},{2}]".format(position[0],position[1],position[2]))
                # print(" position : " + str(position[0]))
                obj_detected= False

                for obj in objects.object_list:
                    # bounding_box = first_object.bounding_box_2d               
                    # # Format top left - 0 , top right - 1 ,bottom right - 2, bottom left - 3,
                    # object distance 
                    # coordinate system -> z points away from camera and y points upward while x points right
                    obj_detected= True
                    print("object detected : ")
                    distance = math.sqrt(obj.position[0]*obj.position[0] + obj.position[2]*obj.position[2])
                    height = obj.position[1]
                    bounding_box = obj.bounding_box_2d 
                    bbox = np.concatenate((bounding_box[0], bounding_box[2]), axis=0)

                    box = np.array([obj.position[0], obj.position[2],obj.velocity[0],obj.velocity[2],0,0]) # set acc to zero will be calculated later
                    #print(" box : ",box)
                    key = obj.id
                    velocity_is_nan = math.isnan(obj.velocity[0])
                    frame_rate = zed.get_camera_information().camera_fps
                    #print("Camera FPS: {0}.".format(frame_rate))
                    #print(" velocity_is_nan : ",velocity_is_nan)
                    if key in objects_dic:  # check if key is already saved in dictionary
                        objects_dic[key].add_box(box)
                        objects_dic[key].distance = distance
                        objects_dic[key].height = height
                        objects_dic[key].bounding = bbox
                        batch_dic[key] =  objects_dic[key]
                        # print("--------------------------------Person object old!--------------------------------")
                        # print (batch_dic[key].history)
                    elif (not velocity_is_nan):
                        new_person = person(key,box,bbox,distance,height,frame_rate)
                        batch_dic[key] = new_person
                        # print(" Person "+str(key)+" added!")
                        # print("--------------------------------Person object!--------------------------------")
                        # print (batch_dic[key].history)
                objects_dic = batch_dic
        obj_added = False        
        for dic_key in objects_dic:
            # print("ID: " + str(dic_key) + "\n")
            # print(objects_dic[dic_key].history)
            if (objects_dic[dic_key].index < 5):             # wait at least until 4 frames of a person are detected..index decreases starting  from 8
                # prediction_input.append(objects_dic[dic_key].history)
                # print(" obj before : ",objects_dic[dic_key].history)
                              ##############################################
                prediction_Xglobal.append(objects_dic[dic_key].history)
                rel_state = np.zeros_like(objects_dic[dic_key].history[0])
                rel_state[0:2] = np.array(objects_dic[dic_key].history)[-1, 0:2]
                mean = rel_state
                std = [3, 3, 2 ,2, 1, 1]
                normalized = np.where(np.isnan(objects_dic[dic_key].history), np.array(np.nan), (objects_dic[dic_key].history - mean) / std)
                # print(" obj After : ",normalized)
                prediction_input.append(normalized)
                prediction_input_dist.append(objects_dic[dic_key].distance)
                prediction_input_height.append(objects_dic[dic_key].height)
                bbox_list.append(objects_dic[dic_key].bounding)

                obj_added= True
                #print(" distance From pred : " + str(objects_dic[dic_key].distance))
                
        
        #print("Predictions Input: " + str(prediction_input))
        prediction_input=torch.FloatTensor(prediction_input)
        prediction_Xglobal=torch.FloatTensor(prediction_Xglobal)
        #print("Predictions Input: " ,prediction_input)

        image_ocv = image.get_data() 
        if obj_detected and obj_added: # obj_detected:
            X_global = prediction_Xglobal.to(device)
            input_x = prediction_input.to(device)

            neighbors_st, adjacency, first_history_indices = None, None, None
            print("Input : " ,(prediction_Xglobal))
            #print("-----------------  prediction_input npormalized: " ,(prediction_input[0]))

           


            pred_goal, pred_traj, _, dist_goal, dist_traj = model(input_x, 
                                                                neighbors_st=neighbors_st,
                                                                adjacency=adjacency,
                                                                z_mode=False, 
                                                                cur_pos=X_global[:, -1, :cfg.MODEL.DEC_OUTPUT_DIM],
                                                                first_history_indices=first_history_indices)


            index_batch = index_batch + 1 
            df_sample_pred_before = pred_traj.detach().to('cpu').numpy() #.cpu().numpy() 
            final_prediction_before = np.array(df_sample_pred_before)
            final_prediction_before= np.swapaxes(final_prediction_before,1,2)
            # print("-----------------  final_prediction Array Shape  before: " ,(final_prediction_before.shape))

           
            # transfer back to global coordinates
            ret = post_process(cfg, X_global, X_global, pred_traj, pred_goal=pred_goal, dist_traj=dist_traj, dist_goal=dist_goal)
            X_global, y_global, pred_goal, pred_traj, dist_traj, dist_goal = ret
            #
            #print("----------------- prediction: " ,(final_prediction_before[0]))

            final_prediction = np.array(pred_traj)

            final_prediction= np.swapaxes(final_prediction,1,2)

            for pred,y,bbox,dist in zip (final_prediction,prediction_input_height,bbox_list,prediction_input_dist):
                # for i in pred[0]:
                #     # print("------------------------I is --------",(i[0]), " ",(i[1]),"--------------------------------")
                #     x=i[0]
                #     z=i[1]
                #     calibration_params = zed.get_camera_information().camera_configuration.calibration_parameters
                #     fx = calibration_params.left_cam.fx
                #     fy = calibration_params.left_cam.fy
                #     cx = calibration_params.left_cam.cx
                #     cy = calibration_params.left_cam.cy
                #     pixelX = cx + (x*fx) / z;
                #     pixelY = cy + (y*fy) / z;
                #     centerOfCircle = (int(pixelX),int(pixelY))
                #     thickness = -1
                #     color = (255, 0, 0)
                #     radius=8

                #     #cv2.circle(image_ocv, centerOfCircle,radius , [0, 0, 255], thickness)
                # check if all x-numbers are on the same side( this means the z-axis which we can assume is y in our case has been crossed)
                # if there is sign change in x-axis it means a person is going to cross through the center of the camera's view
                start_point = (int(bbox[0]),int(bbox[1]))
                end_point = (int(bbox[2]),int(bbox[3]))
                print("----------------- prediction: " ,( pred[0]))
                count = sum(n[0] < 0 for n in pred[0])
                print("count: ",count)
                if (dist < cfg_zed['safe_distance']  ):
                    print(" Dangerously Close to Vehicle!")
                    cv2.rectangle(image_ocv, start_point, end_point,(cfg_zed['danger_color']), 1)
                elif (count!=0 and count!=12 and (dist < cfg_zed['max_cross_detection'])) :
                     print("Crossing!",dis)
                     cv2.rectangle(image_ocv, start_point, end_point,(cfg_zed['crossing_color']), 1)
                else:
                     print("SAFE")
                     cv2.rectangle(image_ocv, start_point, end_point, (cfg_zed['safe_color']), 1) 

                zed.get_position(zed_pose, sl.REFERENCE_FRAME.CAMERA)
                # Display translation and timestamp
                py_translation = sl.Translation()
                tx = round(zed_pose.get_translation(py_translation).get()[0], 3)
                ty = round(zed_pose.get_translation(py_translation).get()[1], 3)
                tz = round(zed_pose.get_translation(py_translation).get()[2], 3)
                #print("Translation: tx: {0}, ty:  {1}, tz:  {2}, timestamp: {3}\n".format(tx, ty, tz, zed_pose.timestamp)) # zed_pose.timestamp
        
                #print("--------------- camera position ----------------------: ",camera_position)
                #print("--------------pred ------------------",pred[0])
        cv2.imshow("Image", image_ocv)
        cv2.waitKey(1)
    image.free(memory_type=sl.MEM.CPU)
    zed.disable_object_detection()
    zed.disable_positional_tracking()
    cv2.destroyAllWindows()
    zed.close()

def inference_kde_nll(cfg, epoch, model, dataloader, device, logger=None):
    model.eval()
    all_pred_goals = []
    all_gt_goals = []
    all_pred_trajs = []
    all_gt_trajs = []
    all_kde_nll = []
    all_per_step_kde_nll = []
    num_samples = model.K
    model.K = 2000
    with torch.set_grad_enabled(False):
        for iters, batch in enumerate(tqdm(dataloader), start=1):
            X_global = batch['input_x'].to(device)
            y_global = batch['target_y']
            img_path = batch['cur_image_file']
            resolution = batch['pred_resolution'].numpy()
            # For ETH_UCY dataset only
            if cfg.DATASET.NAME in ['eth', 'hotel', 'univ', 'zara1', 'zara2']:
                input_x = batch['input_x_st'].to(device)
                neighbors_st = restore(batch['neighbors_x_st'])
                adjacency = restore(batch['neighbors_adjacency'])
                first_history_indices = batch['first_history_index']
            else:
                input_x = X_global
                neighbors_st, adjacency, first_history_indices = None, None, None
            
            pred_goal, pred_traj, _, _, _ = model(input_x, 
                                                    neighbors_st=neighbors_st,
                                                    adjacency=adjacency,
                                                    z_mode=False, 
                                                    cur_pos=X_global[:, -1, :cfg.MODEL.DEC_OUTPUT_DIM],
                                                    first_history_indices=first_history_indices)
            
            # transfer back to global coordinates
            ret = post_process(cfg, X_global, y_global, pred_traj, pred_goal=pred_goal, dist_traj=None, dist_goal=None)
            X_global, y_global, pred_goal, pred_traj, dist_traj, dist_goal = ret
            for i in range(len(pred_traj)):
                KDE_NLL, KDE_NLL_PER_STEP = compute_kde_nll(pred_traj[i:i+1], y_global[i:i+1])
                all_kde_nll.append(KDE_NLL)
                all_per_step_kde_nll.append(KDE_NLL_PER_STEP)
        KDE_NLL = np.array(all_kde_nll).mean()
        KDE_NLL_PER_STEP = np.stack(all_per_step_kde_nll, axis=0).mean(axis=0)
        # Evaluate
        Goal_NLL = KDE_NLL_PER_STEP[-1]
        nll_dict = {'KDE_NLL': KDE_NLL} if cfg.MODEL.LATENT_DIST == 'categorical' else {'KDE_NLL': KDE_NLL, 'Goal_NLL': Goal_NLL}
        info = "Testing prediction KDE_NLL:{:.4f}, per step NLL:{}".format(KDE_NLL, KDE_NLL_PER_STEP)
        if hasattr(logger, 'log_values'):
            logger.info(info)
        else:
            print(info)
        if hasattr(logger, 'log_values'):
            logger.log_values(nll_dict)

    # reset model.K back to 20
    model.K = num_samples
    return KDE_NLL
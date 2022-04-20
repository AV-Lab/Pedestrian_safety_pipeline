import numpy as np
x=[[1,2,3,4,5,6],
    [7,8,9,10,11,12],
    [13,14,15,16,17,18],
    [19,20,21,22,23,24]]





input  = [[ 0.02666667 , 0.11333334 , 0.2125   ,  -0.5375   ,   1.125   ,   -1.8125    ],
 [ 0.02333333 , 0.07 ,      -0.0125 ,    -0.1625,     -1.125 ,      1.875     ],
 [-0.03   ,     0.08333334, -0.2   ,      0.05  ,     -0.9375  ,    1.0625    ],
 [-0.07333333,  0.05333333, -0.1625 ,    -0.1125,      0.1875 ,    -0.8125    ],
 [-0.10333333,  0.01333333, -0.1125 ,    -0.15  ,      0.25    ,   -0.1875    ],
 [-0.07333333, -0.00666667,  0.1125 ,    -0.075 ,   1.125    ,   0.375     ],
 [-0.01333333, -0.00666667,  0.225  ,     0.      ,    0.5625   ,   0.375     ],
 [ 0.   ,       0.   ,       0.05  ,      0.025   ,   -0.875  ,     0.125     ]]
Xglobal =   [[ 7.4064655, -1.3946358 , 0.425,     -1.075 ,     1.125 ,    -1.8125,   ],
 [ 7.396466, -1.5246358 ,-0.025  ,   -0.325  ,   -1.125  ,    1.875    ],
 [ 7.236466,  -1.4846358 ,-0.4    ,    0.1    ,   -0.9375  ,   1.0625   ],
 [ 7.106466,  -1.5746359 ,-0.325   ,  -0.225  ,    0.1875 ,   -0.8125   ],
 [ 7.0164657 ,-1.6946359, -0.225  ,   -0.3    ,    0.25   ,   -0.1875   ],
 [ 7.106466,  -1.7546358 , 0.225  ,   -0.15   ,    1.125   ,  0.375    ],
 [ 7.2864656, -1.7546358 , 0.45  ,     0.    ,     0.5625  ,   0.375    ],
 [ 7.3264656, -1.7346358 , 0.1    ,    0.05  ,    -0.875   ,   0.125    ]]

rel_state = np.zeros_like(Xglobal[0])
print("rel_state before: ",rel_state)
rel_state[0:2] = np.array(Xglobal)[-1, 0:2]
mean = rel_state
print("rel_state After: ",rel_state)
std = [3, 3, 2 ,2, 1, 1]

changed = np.where(np.isnan(Xglobal), np.array(np.nan), (Xglobal - mean) / std)
print(" changed : ",changed)
# import sys
# import ogl_viewer.viewer as gl
# import pyzed.sl as sl
# import cv2
# import math
# sensors_data = sl.SensorsData()
# # Create a Camera object
# zed = sl.Camera()

# # Create a InitParameters object and set configuration parameters
# init_params = sl.InitParameters()
# init_params.camera_resolution = sl.RESOLUTION.HD1080  # Use HD1080 video mode    
# init_params.coordinate_units = sl.UNIT.METER
# init_params.camera_fps = 30                          # Set fps at 30
# init_params.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP

# import sys
# import ogl_viewer.viewer as gl
# import pyzed.sl as sl
# import cv2
# import math
# sensors_data = sl.SensorsData()
# # Create a Camera object
# zed = sl.Camera()
# # Open the camera
# err = zed.open(init_params)
# if err != sl.ERROR_CODE.SUCCESS:
#     exit(1)
# # Grab new frames and retrieve sensors data
# while zed.grab() == sl.ERROR_CODE.SUCCESS :
#   zed.get_sensors_data(sensors_data, sl.TIME_REFERENCE.IMAGE) # Retrieve only frame synchronized data

#   # Extract IMU data
#   zed_imu = sensors_data.get_imu_data()
#   zed_imu_pose = sl.Transform()


#   ox = round(zed_imu.get_pose(zed_imu_pose).get_orientation().get()[0], 3)
#   oy = round(zed_imu.get_pose(zed_imu_pose).get_orientation().get()[1], 3)
#   oz = round(zed_imu.get_pose(zed_imu_pose).get_orientation().get()[2], 3)
#   ow = round(zed_imu.get_pose(zed_imu_pose).get_orientation().get()[3], 3)

#   # Retrieve linear acceleration and angular velocity
#   linear_acceleration = zed_imu.get_linear_acceleration()
#   angular_velocity = zed_imu.get_angular_velocity()
#   quaternion = zed_imu.get_pose().get_orientation().get()
#   print("IMU Orientation: {}".format(quaternion))

#   print("IMU Orientation: Ox: {0}, Oy: {1}, Oz {2}, Ow: {3}\n".format(ox, oy, oz, ow))
#   #print("POSE : ",pose)
  
#     # cv2.destroyAllWindows(),linear_acceleration
# zed.close()

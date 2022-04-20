# For indepth information about the trajectory prediction procedure see the documentation on README_OLD.md

# Process (inference in trainer.py):

- step 1: detect pedestrians via zed camera
- step 2: save the postion and velocity of each pedestrian and save history of the last 8 data points (save the bounding box and distance from the camera as well)
- step 3: calculate the acceleration of each pedestrian and add it to the person history
- step 4: Each time a person is detected in 3 or more consecutive frames, predict future trajectory
- step 5: Since the prediction is multimodal, select one (usually first prediction)
- step 6: if the distance between the camera and the person is smaller than the threshold for safety distance --> RED bounding box is shown
else if the person is not crossing the path of the camera and  is in the range of max_detection for crossing ---> Blue bounding box is shown
if neither of teh above cases are true then it is assumed that the person is safe ----> Green bounding

path-crossing : we assume a person is crossing the path of the camera if the predicted trajectory is passes through the straight path of the camera (z-axis for zed camera)
max_cross_detection : is the max distance to consider a person as crossing the path. Beyond this point even if a person is crossing, they are considered to be safe.

max_cross_detection and other parameters such as threshold for safe distance, frame rate, color of the bounding boxs can be changed from config_latest.yml

 

# ENV SET UP
TO set up the enironment for the system use the following code:
conda env create -f environment.yml

# TO change config files:
config_latest.yml

# To RUN the code use the following command:
python test.py




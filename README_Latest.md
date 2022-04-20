# ETH in use: python test.py --config_file configs/bitrap_np_ETH.yml  CKPT_DIR ETH/bitrap_np_eth.pth

# hyperparams['minimum_history_length'] = cfg.MODEL.MIN_HIST_LEN #1 # different from trajectron++, we don't use short histories.
hyperparams['state'] = {'PEDESTRIAN':{'position':['x','y'], 'velocity':['x','y'], 'acceleration':['x','y']}}
hyperparams['pred_state'] = {'PEDESTRIAN':{'position':['x','y']}}


# The Eth dataset is manipulated before it is used as input
-> First  each x and y -> x=x-x.mean() and y=y-y.mean()
-> later 4 more variables are added vx vy ax and ay represneting the velocity and accelaration
-> vx = (x(i+1)-x(i) )/ dt  they take dt to be 0.4 similarly
-> ax = (v(i+1)-v(i) )/ dt   this is calculated if acc is not supplied as input 

# point based trajectory : input length is 8 datapoints with  6 parameters each {'position':['x','y'], 'velocity':['x','y'], 'acceleration':['x','y']} ..format [xp,yp,xv,yv,xa,ya] p-position , v-velocity and a-acceleration 


# ENV SET UP
TO set up the enironment for the system use the following code:
conda env create -f environment.yml

# To RUN the code use the following command:
python test.py
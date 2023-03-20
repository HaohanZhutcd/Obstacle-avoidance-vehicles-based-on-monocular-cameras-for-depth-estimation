import numpy as np 
import cv2
import matplotlib
import matplotlib.pyplot as plt


# Check for left and right camera IDs
# These values can change depending on the system
#CamL_id = 2 # Camera ID for left camera
#CamR_id = 0 # Camera ID for right camera

#CamL= cv2.VideoCapture(CamL_id)
#CamR= cv2.VideoCapture(CamR_id)
CamL_id = "/home/pi/car2/data/L/1.jpg"
CamR_id = "/home/pi/car2/data/R/1.jpg"

imgL= cv2.imread(CamL_id)
imgR= cv2.imread(CamR_id)
# Reading the mapping values for stereo image rectification
cv_file = cv2.FileStorage("stereo_rectify_maps.xml", cv2.FILE_STORAGE_READ)
Left_Stereo_Map_x = cv_file.getNode("Left_Stereo_Map_x").mat()
Left_Stereo_Map_y = cv_file.getNode("Left_Stereo_Map_y").mat()
Right_Stereo_Map_x = cv_file.getNode("Right_Stereo_Map_x").mat()
Right_Stereo_Map_y = cv_file.getNode("Right_Stereo_Map_y").mat()
cv_file.release()

# These parameters can vary according to the setup
# Keeping the target object at max_dist we store disparity values
# after every sample_delta distance.
max_dist = 60 # max distance to keep the target object (in cm)
min_dist = 15 # Minimum distance the stereo setup can measure (in cm)
sample_delta = 5 # Distance between two sampling points (in cm)

Z = max_dist 
Value_pairs = []

disp_map = np.zeros((600,600,3))


# Reading the stored the StereoBM parameters
cv_file = cv2.FileStorage("/home/pi/car2/data/depth_estmation_params_py.xml", cv2.FILE_STORAGE_READ)
minDisparity = int(cv_file.getNode("minDisparity").real())
numDisparities = int(cv_file.getNode("numDisparities").real())
blockSize = int(cv_file.getNode("blockSize").real())
uniquenessRatio = int(cv_file.getNode("uniquenessRatio").real())
speckleRange = int(cv_file.getNode("speckleRange").real())
speckleWindowSize = int(cv_file.getNode("speckleWindowSize").real())
disp12MaxDiff = int(cv_file.getNode("disp12MaxDiff").real())
M = cv_file.getNode("M").real()
cv_file.release()

# Defining callback functions for mouse events
def mouse_click(event,x,y,flags,param):
    global Z
    if event == cv2.EVENT_LBUTTONDBLCLK:
        if disparity[y,x] > 0:
            Value_pairs.append([Z,disparity[y,x]])
            print("Distance: %r cm  | Disparity: %r"%(Z,disparity[y,x]))
            Z-=sample_delta



cv2.namedWindow('disp',cv2.WINDOW_NORMAL)
cv2.resizeWindow('disp',640,480)
cv2.namedWindow('left image',cv2.WINDOW_NORMAL)
cv2.resizeWindow('left image',640,480)
cv2.setMouseCallback('disp',mouse_click)

# Creating an object of StereoBM algorithm


imgR_gray = cv2.cvtColor(imgR,cv2.COLOR_BGR2GRAY)
imgL_gray = cv2.cvtColor(imgL,cv2.COLOR_BGR2GRAY)

# Applying stereo image rectification on the left image
Left_nice= cv2.remap(imgL_gray,
                     Left_Stereo_Map_x,
                     Left_Stereo_Map_y,
                     cv2.INTER_LANCZOS4,
                     cv2.BORDER_CONSTANT,
                     0)

# Applying stereo image rectification on the right image
Right_nice= cv2.remap(imgR_gray,
					  Right_Stereo_Map_x,
                      Right_Stereo_Map_y,
					  cv2.INTER_LANCZOS4,
					  cv2.BORDER_CONSTANT,
					  0)

stereo = cv2.StereoSGBM_create(
        minDisparity = minDisparity,
        numDisparities = numDisparities,
        blockSize = blockSize,
        #P1 = P1,
        #P2 = P2,
        disp12MaxDiff = disp12MaxDiff,
        uniquenessRatio = uniquenessRatio,
        speckleWindowSize = speckleWindowSize,
        speckleRange = speckleRange
    )
# Calculating disparity using the StereoBM algorithm
disparity = stereo.compute(Left_nice,Right_nice).astype(np.float32)

# Calculating disparith using the StereoSGBM algorithm
disparity = cv2.normalize(disparity,0,255,cv2.NORM_MINMAX)

# Displaying the disparity map
cv2.imshow("disp",disparity)
cv2.imshow("left image",imgL)
cv2.waitKey(0)
if Z < min_dist:
    pass


# solving for M in the following equation
# ||    depth = M * (1/disparity)   ||
# for N data points coeff is Nx2 matrix with values 
# 1/disparity, 1
# and depth is Nx1 matrix with depth values

value_pairs = np.array(Value_pairs)
z = value_pairs[:,0]
disp = value_pairs[:,1]
disp_inv = 1/disp
'''
# Plotting the relation depth and corresponding disparity
fig, (ax1,ax2) = plt.subplots(1,2,figsize=(12,6))
ax1.plot(disp, z, 'o-')
ax1.set(xlabel='Normalized disparity value', ylabel='Depth from camera (cm)',
       title='Relation between depth \n and corresponding disparity')
ax1.grid()
ax2.plot(disp_inv, z, 'o-')
ax2.set(xlabel='Inverse disparity value (1/disp) ', ylabel='Depth from camera (cm)',
       title='Relation between depth \n and corresponding inverse disparity')
ax2.grid()
plt.show()
'''

# Solving for M using least square fitting with QR decomposition method
coeff = np.vstack([disp_inv, np.ones(len(disp_inv))]).T
ret, sol = cv2.solve(coeff,z,flags=cv2.DECOMP_QR)
M = sol[0,0]
C = sol[1,0]
print("Value of M = ",M)


# Storing the updated value of M along with the stereo parameters
cv_file = cv2.FileStorage("../data/depth_estmation_params_py.xml", cv2.FILE_STORAGE_WRITE)
cv_file.write("minDisparity",minDisparity)
cv_file.write("numDisparities",numDisparities)
cv_file.write("blockSize",blockSize)
cv_file.write("disp12MaxDiff",disp12MaxDiff)
cv_file.write("uniquenessRatio",uniquenessRatio)
cv_file.write("speckleWindowSize",speckleWindowSize)
cv_file.write("speckleRange",speckleRange)
cv_file.write("M",M)
cv_file.release()

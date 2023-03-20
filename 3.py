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
disparity = None
depth_map = None
# These parameters can vary according to the setup
# Keeping the target object at max_dist we store disparity values
# after every sample_delta distance.
max_depth = 60 # max distance to keep the target object (in cm)
min_depth = 15 # Minimum distance the stereo setup can measure (in cm)
#sample_delta = 5 # Distance between two sampling points (in cm)
depth_thresh = 20



#disp_map = np.zeros((600,600,3))


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
        print("Distance = %.2f cm"%depth_map[y,x])
cv2.namedWindow('disp',cv2.WINDOW_NORMAL)
cv2.resizeWindow('disp',600,600)
cv2.setMouseCallback('disp',mouse_click)

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

output_canvas = None

def obstacle_avoid():

	# Mask to segment regions with depth less than threshold
	mask = cv2.inRange(depth_map,10,depth_thresh)

	# Check if a significantly large obstacle is present and filter out smaller noisy regions
	if np.sum(mask)/255.0 > 0.01*mask.shape[0]*mask.shape[1]:

		# Contour detection 
		contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
		cnts = sorted(contours, key=cv2.contourArea, reverse=True)
		
		# Check if detected contour is significantly large (to avoid multiple tiny regions)
		if cv2.contourArea(cnts[0]) > 0.01*mask.shape[0]*mask.shape[1]:

			x,y,w,h = cv2.boundingRect(cnts[0])

			# finding average depth of region represented by the largest contour 
			mask2 = np.zeros_like(mask)
			cv2.drawContours(mask2, cnts, 0, (255), -1)

			# Calculating the average depth of the object closer than the safe distance
			depth_mean, _ = cv2.meanStdDev(depth_map, mask=mask2)
			
			# Display warning text
			cv2.putText(output_canvas, "WARNING !", (x+5,y-40), 1, 2, (0,0,255), 2, 2)
			cv2.putText(output_canvas, "Object at", (x+5,y), 1, 2, (100,10,25), 2, 2)
			cv2.putText(output_canvas, "%.2f cm"%depth_mean, (x+5,y+40), 1, 2, (100,10,25), 2, 2)

	else:
		cv2.putText(output_canvas, "SAFE!", (100,100),1,3,(0,255,0),2,3)

	cv2.imshow('output_canvas',output_canvas)



# Creating an object of StereoBM algorithm

output_canvas = imgL.copy()
imgL_gray = cv2.imread(CamL_id, cv2.IMREAD_GRAYSCALE)
imgR_gray = cv2.imread(CamR_id, cv2.IMREAD_GRAYSCALE)

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


# Calculating disparity using the StereoBM algorithm
disparity = stereo.compute(Left_nice,Right_nice).astype(np.float32)

# Calculating disparith using the StereoSGBM algorithm
disparity = cv2.normalize(disparity,0,255,cv2.NORM_MINMAX)
depth_map = M/(disparity)
mask_temp = cv2.inRange(depth_map,min_depth,max_depth)
depth_map = cv2.bitwise_and(depth_map,depth_map,mask=mask_temp)
obstacle_avoid()
# Displaying the disparity map
cv2.imshow("disp",disparity)
cv2.imshow("left image",imgL)
cv2.waitKey(0)
if Z < min_dist:
    pass




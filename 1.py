import numpy as np 
import cv2
minDisparity = 0
numDisparities = 10*16
blockSize = 10 * 2 + 1
disp12MaxDiff = 40
uniquenessRatio = 15
speckleWindowSize = 100
speckleRange = 1
P1 = 8 * blockSize * blockSize
P2 = 32 * blockSize * blockSize

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

def nothing(x):
    pass

cv2.namedWindow('disp',cv2.WINDOW_NORMAL)
cv2.resizeWindow('disp',640,480)



# Creating an object of StereoBM algorithm
#stereo = cv2.StereoBM_create()

'''
CamL_id = "/home/pi/car2/data/L/1.jpg"
CamR_id = "/home/pi/car2/data/R/1.jpg"
imgL= cv2.imread(CamL_id)
imgR= cv2.imread(CamR_id)
'''
#imgL_gray = cv2.imread(CamL_id, cv2.IMREAD_GRAYSCALE)
#imgR_gray = cv2.imread(CamR_id, cv2.IMREAD_GRAYSCALE)
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


disparity = stereo.compute(Left_nice,Right_nice).astype(np.float32)

# Calculating disparith using the StereoSGBM algorithm
disparity = (disparity/16.0 - minDisparity)/numDisparities
# Calculating disparity using the StereoBM algorithm
#disparity = stereo.compute(Left_nice,Right_nice)
# NOTE: compute returns a 16bit signed single channel image,
# CV_16S containing a disparity map scaled by 16. Hence it 
# is essential to convert it to CV_32F and scale it down 16 times.

# Converting to float32 
#disparity = disparity.astype(np.float32)

# Scaling down the disparity values and normalizing them 
#disparity = (disparity/16.0 - minDisparity)/numDisparities

# Displaying the disparity map
cv2.imshow("disp",disparity)
cv2.waitKey(0)
#cv2.destroyAllWindows()

print("Saving depth estimation paraeters ......")

cv_file = cv2.FileStorage("/home/pi/car2/data/depth_estmation_params_py.xml", cv2.FILE_STORAGE_WRITE)
cv_file.write("minDisparity",minDisparity)
cv_file.write("numDisparities",numDisparities)
cv_file.write("blockSize",blockSize)
cv_file.write("disp12MaxDiff",disp12MaxDiff)
cv_file.write("uniquenessRatio",uniquenessRatio)
cv_file.write("speckleWindowSize",speckleWindowSize)
cv_file.write("speckleRange",speckleRange)
cv_file.write("P1",P1)
cv_file.write("P2",P2)
cv_file.write("M",39.075)
cv_file.release()

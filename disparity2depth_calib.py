from picar import front_wheels, back_wheels
from picar.SunFounder_PCA9685 import Servo
import picar
import time
import cv2
import numpy as np
import picar
import os
picar.setup()
db_file = "/home/pi/small_car/SunFounder_PiCar-V/remote_control/remote_control/driver/config"
fw = front_wheels.Front_Wheels(debug=False, db=db_file)
bw = back_wheels.Back_Wheels(debug=False, db=db_file)
pan_servo = Servo.Servo(1)
tilt_servo = Servo.Servo(2)
bw.ready()
fw.ready()
pan_servo.write(90)
tilt_servo.write(90)
# Reading the mapping values for stereo image rectification
cv_file = cv2.FileStorage("stereo_rectify_maps.xml", cv2.FILE_STORAGE_READ)
Left_Stereo_Map_x = cv_file.getNode("Left_Stereo_Map_x").mat()
Left_Stereo_Map_y = cv_file.getNode("Left_Stereo_Map_y").mat()
Right_Stereo_Map_x = cv_file.getNode("Right_Stereo_Map_x").mat()
Right_Stereo_Map_y = cv_file.getNode("Right_Stereo_Map_y").mat()
cv_file.release()
def left_image(left_detection):
    if left_detection == True:
        pan_angle = 95
        tilt_angle = 90
        pan_servo.write(pan_angle)
        tilt_servo.write(tilt_angle)
        
        capture_duration = 5
        cap = cv2.VideoCapture(0)
        left_fourcc = cv2.VideoWriter_fourcc(*'XVID')
        left_out = cv2.VideoWriter('/home/pi/car2/data/video/L/left.avi', left_fourcc, 30.0, (640,480))
        
        start_time = time.time()
        #while True:
        while(int(time.time() - start_time) < capture_duration):
            ret,frame_left = cap.read()
            if not ret:
                print("Not ret")
                break
            timer = capture_duration - int(time.time() - start_time)
            imgL_temp = frame_left.copy()
            #cv2.putText(imgL_temp,"%r"%timer,(50,50),1,5,(55,0,0),5)
            left_out.write(imgL_temp)
            cv2.imshow("left_orig",frame_left)
            cv2.waitKey(2)
            #if cv2.waitKey(1) & 0xFF == ord("q"):
            #    break
        cap.release()
        left_out.release()
        cv2.destroyAllWindows()
        #time.sleep(0.5)
    else:
        pass

def right_image(right_detection):
    if right_detection == True:
        pan_angle = 85
        tilt_angle = 90
        pan_servo.write(pan_angle)
        tilt_servo.write(tilt_angle)
        
        capture_duration = 5
        cap = cv2.VideoCapture(0)
        right_fourcc = cv2.VideoWriter_fourcc(*'XVID')
        right_out = cv2.VideoWriter('/home/pi/car2/data/video/R/right.avi', right_fourcc, 30.0, (640,480))
        start_time = time.time()
        while(cap.isOpened):
        #while(int(time.time() - start_time) < capture_duration):
            ret,frame_right = cap.read()
            if not ret:
                print("Not ret")
                break
            #timer = capture_duration - int(time.time() - start_time)
            #imgR_temp = frame_right.copy()
            #cv2.putText(imgR_temp,"%r"%timer,(50,50),1,5,(55,0,0),5)
            right_out.write(frame_right)
            cv2.imshow("right_orig",frame_right)
            cv2.waitKey(2)
            #if cv2.waitKey(1) & 0xFF == ord("q"):
            #    break
            if time.time() - start_time > capture_duration:
                break
        cap.release()
        right_out.release()
        cv2.destroyAllWindows()
        #time.sleep(0.5)
    else:
        pass
# These parameters can vary according to the setup
# Keeping the target object at max_dist we store disparity values
# after every sample_delta distance.
max_dist = 50 # max distance to keep the target object (in cm)
min_dist = 10 # Minimum distance the stereo setup can measure (in cm)
sample_delta = 5 # Distance between two sampling points (in cm)

Z = max_dist 
Value_pairs = []

disp_map = np.zeros((600,600,3))


# Reading the stored the StereoBM parameters
cv_file = cv2.FileStorage("/home/pi/car2/data/depth_estmation_params_py.xml", cv2.FILE_STORAGE_READ)
'''
numDisparities = int(cv_file.getNode("numDisparities").real())
blockSize = int(cv_file.getNode("blockSize").real())

preFilterType = int(cv_file.getNode("preFilterType").real())
preFilterSize = int(cv_file.getNode("preFilterSize").real())
preFilterCap = int(cv_file.getNode("preFilterCap").real())
textureThreshold = int(cv_file.getNode("textureThreshold").real())
uniquenessRatio = int(cv_file.getNode("uniquenessRatio").real())
speckleRange = int(cv_file.getNode("speckleRange").real())
speckleWindowSize = int(cv_file.getNode("speckleWindowSize").real())
disp12MaxDiff = int(cv_file.getNode("disp12MaxDiff").real())
minDisparity = int(cv_file.getNode("minDisparity").real())
'''
M = cv_file.getNode("M").real()
cv_file.release()

# Defining callback functions for mouse events
def mouse_click(event,x,y,flags,param):
    global Z
    if event == cv2.EVENT_LBUTTONDBLCLK:
        #print("click successfully!")
        if disparity[y,x] > 0:
            Value_pairs.append([Z,disparity[y,x]])
            print("Distance: %r cm  | Disparity: %r"%(Z,disparity[y,x]))
            Z-=sample_delta



cv2.namedWindow('disp',cv2.WINDOW_NORMAL)
cv2.resizeWindow('disp',600,600)
cv2.namedWindow('left image',cv2.WINDOW_NORMAL)
cv2.resizeWindow('left image',600,600)
cv2.setMouseCallback('disp',mouse_click)

cv2.createTrackbar('num','disp',9,15, lambda x: None)
cv2.createTrackbar('blockSize','disp',15,255,lambda x: None)

# Creating an object of StereoBM algorithm
stereo = cv2.StereoBM_create()
print("begin to capture image")
'''
time.sleep(0.1)
left_image(left_detection= True)
time.sleep(0.1)
right_image(right_detection= True)
time.sleep(0.1)
pan_servo.write(90)
tilt_servo.write(90)
'''
print("finish capture and init the camera angel")
#capL = cv2.VideoCapture(0)
#capR = cv2.VideoCapture(0)
print("read the image captured")
capL = cv2.VideoCapture("/home/pi/car2/data/video/L/left.avi")
capR = cv2.VideoCapture("/home/pi/car2/data/video/R/right.avi")
while True:
    retL,frameL = capL.read()
    retR,frameR = capR.read()
    if retL and retR:
        imgR_gray = cv2.cvtColor(frameR,cv2.COLOR_BGR2GRAY)
        imgL_gray = cv2.cvtColor(frameL,cv2.COLOR_BGR2GRAY)
        
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
        
        num = cv2.getTrackbarPos('num','disp')
        blockSize = cv2.getTrackbarPos('blockSize','disp')
        numDisparities = 16*num
        if blockSize % 2 == 0:
            blockSize += 1
        if blockSize <5:
            blockSize = 5
        stereo = cv2.StereoBM_create(numDisparities = 16*num,blockSize = 31)
        '''
        
        # Setting the updated parameters before computing disparity map
        stereo.setNumDisparities(numDisparities)
        stereo.setBlockSize(blockSize)
        
        stereo.setPreFilterType(preFilterType)
        stereo.setPreFilterSize(preFilterSize)
        stereo.setPreFilterCap(preFilterCap)
        stereo.setTextureThreshold(textureThreshold)
        stereo.setUniquenessRatio(uniquenessRatio)
        stereo.setSpeckleRange(speckleRange)
        stereo.setSpeckleWindowSize(speckleWindowSize)
        stereo.setDisp12MaxDiff(disp12MaxDiff)
        stereo.setMinDisparity(minDisparity)
        '''
        # Calculating disparity using the StereoBM algorithm
        disparity = stereo.compute(Left_nice,Right_nice)
        # NOTE: compute returns a 16bit signed single channel image,
        # CV_16S containing a disparity map scaled by 16. Hence it 
        # is essential to convert it to CV_16S and scale it down 16 times.

        # Converting to float32 
        disparity = disparity.astype(np.float32)

        # Scaling down the disparity values and normalizing them 
        disparity = (disparity/16.0 )/numDisparities

        # Displaying the disparity map
        cv2.imshow("disp",disparity)
        cv2.imshow("left image",frameL)
        if cv2.waitKey(20) == 27:
            break
        if Z < min_dist:
            break
    else:
        print("finish")
        break
cv2.destroyAllWindows()


# solving for M in the following equation
# ||    depth = M * (1/disparity)   ||
# for N data points coeff is Nx2 matrix with values 
# 1/disparity, 1
# and depth is Nx1 matrix with depth values

value_pairs = np.array(Value_pairs)
z = value_pairs[:,0]
disp = value_pairs[:,1]
disp_inv = 1/disp



# Solving for M using least square fitting with QR decomposition method
coeff = np.vstack([disp_inv, np.ones(len(disp_inv))]).T
ret, sol = cv2.solve(coeff,z,flags=cv2.DECOMP_QR)
M = sol[0,0]
C = sol[1,0]
print("Value of M = ",M)

'''
# Storing the updated value of M along with the stereo parameters
cv_file = cv2.FileStorage("../data/depth_estmation_params_py.xml", cv2.FILE_STORAGE_WRITE)

cv_file.write("numDisparities",numDisparities)
cv_file.write("blockSize",blockSize)
cv_file.write("preFilterType",preFilterType)
cv_file.write("preFilterSize",preFilterSize)
cv_file.write("preFilterCap",preFilterCap)
cv_file.write("textureThreshold",textureThreshold)
cv_file.write("uniquenessRatio",uniquenessRatio)
cv_file.write("speckleRange",speckleRange)
cv_file.write("speckleWindowSize",speckleWindowSize)
cv_file.write("disp12MaxDiff",disp12MaxDiff)
cv_file.write("minDisparity",minDisparity)

cv_file.write("M",M)
cv_file.release()
'''
print("finish write")
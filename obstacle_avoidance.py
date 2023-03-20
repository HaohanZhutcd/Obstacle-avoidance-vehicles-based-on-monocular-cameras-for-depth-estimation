from picar import front_wheels, back_wheels
from picar.SunFounder_PCA9685 import Servo
import picar
import time
import cv2
import numpy as np
import picar
import os
#from calibrate import Q
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

disparity = None
depth_map = None

# These parameters can vary according to the setup
max_depth = 60 # maximum distance the setup can measure (in cm)
min_depth = 15 # minimum distance the setup can measure (in cm)
depth_thresh = 30 # Threshold for SAFE distance (in cm)

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


# mouse callback function
def mouse_click(event,x,y,flags,param):
    global Z
    if event == cv2.EVENT_LBUTTONDBLCLK:
        print(disparity[y,x])
        if disparity[y,x] > 0:
            depth_map = 14.2556/disparity
            print("Distance = %.2f cm"%depth_map[y,x])


cv2.namedWindow('disp',cv2.WINDOW_NORMAL)
cv2.resizeWindow('disp',600,600)
cv2.setMouseCallback('disp',mouse_click)

cv2.createTrackbar('num','disp',13,30, lambda x: None)
cv2.createTrackbar('blockSize','disp',15,255,lambda x: None)


output_canvas = None

# Creating an object of StereoBM algorithm
stereo = cv2.StereoBM_create()
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
def obstacle_avoid():
    # Mask to segment regions with depth less than threshold
    mask = cv2.inRange(depth_map,5,depth_thresh)

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
'''
time.sleep(0.1)
left_image(left_detection= True)
time.sleep(0.1)
right_image(right_detection= True)
time.sleep(0.1)
pan_servo.write(90)
tilt_servo.write(90)
'''
CamL = cv2.VideoCapture("/home/pi/car2/data/video/L/left.avi")
CamR = cv2.VideoCapture("/home/pi/car2/data/video/R/right.avi")
while True:
    retR, imgR= CamR.read()
    retL, imgL= CamL.read()
    
    if retL and retR:
        
        output_canvas = imgL.copy()

        

        # Applying stereo image rectification on the left image
        Left_nice= cv2.remap(imgL,
                             Left_Stereo_Map_x,
                             Left_Stereo_Map_y,
                             cv2.INTER_LANCZOS4,
                             cv2.BORDER_CONSTANT,
                             0)
        
        # Applying stereo image rectification on the right image
        Right_nice= cv2.remap(imgR,
                              Right_Stereo_Map_x,
                              Right_Stereo_Map_y,
                              cv2.INTER_LANCZOS4,
                              cv2.BORDER_CONSTANT,
                              0)
        
        imgR_gray = cv2.cvtColor(Right_nice,cv2.COLOR_BGR2GRAY)
        imgL_gray = cv2.cvtColor(Left_nice,cv2.COLOR_BGR2GRAY)
        
        
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
        disparity = stereo.compute(imgL_gray,imgR_gray)
        #cv2.imshow("1",disparity)
        disparity = disparity.astype(np.float32)
        #disp = cv2.normalize(disparity, disparity, alpha=0, beta=255,norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        #cv2.imshow("dist",disp)
        # Normalizing the disparity map
        #disp = cv2.normalize(disparity, disparity, alpha=0, beta=255,norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        disparity = (disparity/16.0 )/numDisparities
        #print(disparity[y,x])
        #cv2.imshow("disparity_test", disparity)
        #print("disparity:",disparity )
        #M = 142.556
        #depth_map = 142.556/disparity
        #cv2.imshow("depth_map",depth_map)
        #print("depth_map",depth_map)
        # for depth in (cm)

        #mask_temp = cv2.inRange(depth_map,min_depth,max_depth)
        #depth_map = cv2.bitwise_and(depth_map,depth_map,mask=mask_temp)
        #cv2.imshow("depth",depth_map)
        #obstacle_avoid()
        max_depth = 60 # maximum distance the setup can measure (in cm)
        min_depth = 15 # minimum distance the setup can measure (in cm)
        depth_map = 14.2556/disparity
        cv2.resizeWindow("disp",700,700)
        cv2.imshow("disp",disparity)
        mask_temp = cv2.inRange(depth_map,min_depth,max_depth)
        depth_map = cv2.bitwise_and(depth_map,depth_map,mask=mask_temp)
        cv2.imshow("mask",depth_map)
        obstacle_avoid()
        if cv2.waitKey(20) == 27:
            break
    
    else:
        break
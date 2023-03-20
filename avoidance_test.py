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

# mouse callback function
def mouse_click(event,x,y,flags,param):
    global Z
    if event == cv2.EVENT_LBUTTONDBLCLK:
        print(disparity[y,x])
        if disparity[y,x] > 0:
            depth_map = 22.5/disparity
            print("Distance = %.2f cm"%depth_map[y,x])


cv2.namedWindow('disp',cv2.WINDOW_NORMAL)
cv2.resizeWindow('disp',600,600)
cv2.setMouseCallback('disp',mouse_click)
# Creating an object of StereoBM algorithm

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
cv2.createTrackbar('minDisparity','disp',0,1, lambda x: None)
cv2.createTrackbar('numDisparities','disp',10,16, lambda x: None)
cv2.createTrackbar('blockSize','disp',15,255,lambda x: None)
cv2.createTrackbar('P1','disp',8,20,lambda x: None)
cv2.createTrackbar('P2','disp',32,50,lambda x: None)
cv2.createTrackbar('disp12MaxDiff','disp',1,50,lambda x: None)
cv2.createTrackbar('uniquenessRatio','disp',15,20,lambda x: None)
cv2.createTrackbar('speckleWindowSize','disp',100,200,lambda x: None)
cv2.createTrackbar('speckleRange','disp',2,100,lambda x: None)
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
        cv2.imshow("original left",imgL)
        # Applying stereo image rectification on the left image
        Left_nice= cv2.remap(imgL,
                             Left_Stereo_Map_x,
                             Left_Stereo_Map_y,
                             cv2.INTER_LINEAR
                             )
                             #cv2.INTER_LANCZOS4,
                             #cv2.BORDER_CONSTANT,
                             #0)
        
        # Applying stereo image rectification on the right image
        Right_nice= cv2.remap(imgR,
                              Right_Stereo_Map_x,
                              Right_Stereo_Map_y,
                              cv2.INTER_LINEAR
                              )
                              #cv2.BORDER_CONSTANT,
                              #0)
        
        imgR_gray = cv2.cvtColor(Right_nice,cv2.COLOR_BGR2GRAY)
        imgL_gray = cv2.cvtColor(Left_nice,cv2.COLOR_BGR2GRAY)
        
     
        minDisparity = cv2.getTrackbarPos('minDisparity','disp')
        numDisparities = cv2.getTrackbarPos('numDisparities','disp')*16
        blockSize = cv2.getTrackbarPos('blockSize','disp')
        if blockSize % 2 == 0:
            blockSize += 1
        if blockSize <5:
            blockSize = 5
        P1 = cv2.getTrackbarPos('P1','disp') * blockSize *blockSize
        P2 = cv2.getTrackbarPos('P2','disp') * blockSize *blockSize
        disp12MaxDiff = cv2.getTrackbarPos('disp12MaxDiff','disp')
        uniquenessRatio = cv2.getTrackbarPos('uniquenessRatio','disp')
        speckleWindowSize = cv2.getTrackbarPos('speckleWindowSize','disp')
        speckleRange = cv2.getTrackbarPos('speckleRange','disp')
        
        stereo = cv2.StereoSGBM_create(minDisparity = minDisparity,
                                       numDisparities = numDisparities,
                                       blockSize = blockSize,
                                       P1 = P1,
                                       P2 = P2,
                                       disp12MaxDiff = disp12MaxDiff,
                                       uniquenessRatio = uniquenessRatio,
                                       speckleWindowSize = speckleWindowSize,
                                       speckleRange = speckleRange
                                       )
        
        disparity = stereo.compute(imgL_gray,imgR_gray)
        #cv2.imshow("1",disparity)
        disparity = disparity.astype(np.float32)
        #disparity = cv2.normalize(disparity,0,255,cv2.NORM_MINMAX)
        disparity = (disparity/16.0 )/numDisparities
        M = 142.556
        #depth_map = 142.556/disparity


        #mask_temp = cv2.inRange(depth_map,min_depth,max_depth)
        #depth_map = cv2.bitwise_and(depth_map,depth_map,mask=mask_temp)
        #cv2.imshow("depth",depth_map)
        #obstacle_avoid()

        #cv2.resizeWindow("disp",700,700)
        cv2.imshow("disp",disparity)

        if cv2.waitKey(30) == 27:
            break
    
    else:
        print("video is over")
        break

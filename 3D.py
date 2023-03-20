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
print("Reading parameters ......")
cv_file = cv2.FileStorage("stereo_rectify_maps.xml", cv2.FILE_STORAGE_READ)

Left_Stereo_Map_x = cv_file.getNode("Left_Stereo_Map_x").mat()
Left_Stereo_Map_y = cv_file.getNode("Left_Stereo_Map_y").mat()
Right_Stereo_Map_x = cv_file.getNode("Right_Stereo_Map_x").mat()
Right_Stereo_Map_y = cv_file.getNode("Right_Stereo_Map_y").mat()
cv_file.release()
print("begin to capture image")

time.sleep(0.1)
left_image(left_detection= True)
time.sleep(0.1)
right_image(right_detection= True)
time.sleep(0.1)
pan_servo.write(90)
tilt_servo.write(90)

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

        Left_nice= cv2.remap(frameL,
                             Left_Stereo_Map_x,
                             Left_Stereo_Map_y,
                             cv2.INTER_LANCZOS4,
                             cv2.BORDER_CONSTANT,
                             0)
        Right_nice= cv2.remap(frameR,
                              Right_Stereo_Map_x,
                              Right_Stereo_Map_y,
                              cv2.INTER_LANCZOS4,
                              cv2.BORDER_CONSTANT,
                              0)

        output = Right_nice.copy()
        output[:,:,0] = Right_nice[:,:,0]
        output[:,:,1] = Right_nice[:,:,1]
        output[:,:,2] = Left_nice[:,:,2]

        # output = Left_nice+Right_nice
        output = cv2.resize(output,(700,700))
        cv2.namedWindow("3D movie",cv2.WINDOW_NORMAL)
        cv2.resizeWindow("3D movie",700,700)
        cv2.imshow("3D movie",output)

        cv2.waitKey(0)

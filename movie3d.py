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
#CamL = cv2.VideoCapture("/home/pi/small_car/data/stereoL/img1.png")
#CamR = cv2.VideoCapture("/home/pi/small_car/data/stereoR/img1.png")
CamL = cv2.VideoCapture("/home/pi/small_car/data/left.avi")
CamR = cv2.VideoCapture("/home/pi/small_car/data/right.avi")
print("Reading parameters ......")
cv_file = cv2.FileStorage("/home/pi/car2/data/stereo_rectify_maps.xml", cv2.FILE_STORAGE_READ)

Left_Stereo_Map_x = cv_file.getNode("Left_Stereo_Map_x").mat()
Left_Stereo_Map_y = cv_file.getNode("Left_Stereo_Map_y").mat()
Right_Stereo_Map_x = cv_file.getNode("Right_Stereo_Map_x").mat()
Right_Stereo_Map_y = cv_file.getNode("Right_Stereo_Map_y").mat()
cv_file.release()


while True:
    retR, imgR= CamR.read()
    retL, imgL= CamL.read()
    cv2.imshow("1",imgR)
    #print(str(imgR.shape))
    if retL and retR:
        imgR_gray = cv2.cvtColor(imgR,cv2.COLOR_BGR2GRAY)
        imgL_gray = cv2.cvtColor(imgL,cv2.COLOR_BGR2GRAY)
        print(str(imgL_gray.shape))
        Left_nice= cv2.remap(imgL,
                             Left_Stereo_Map_x,
                             Left_Stereo_Map_y,
                             cv2.INTER_LANCZOS4,
                             cv2.BORDER_CONSTANT, 0)
        Right_nice= cv2.remap(imgR,
                              Right_Stereo_Map_x,
                              Right_Stereo_Map_y,
                              cv2.INTER_LANCZOS4,
                              cv2.BORDER_CONSTANT, 0)

        output = Right_nice.copy()
        output[:,:,0] = Right_nice[:,:,0]
        output[:,:,1] = Right_nice[:,:,1]
        output[:,:,2] = Left_nice[:,:,2]

        # output = Left_nice+Right_nice
        output = cv2.resize(output,(640,480))
        cv2.namedWindow("3D movie",cv2.WINDOW_NORMAL)
        cv2.resizeWindow("3D movie",640,480)
        cv2.imshow("3D movie",output)

        cv2.waitKey(1)

    else:
        break

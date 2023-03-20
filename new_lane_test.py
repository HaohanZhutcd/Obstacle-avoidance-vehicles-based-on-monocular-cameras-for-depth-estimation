from picar import front_wheels, back_wheels
from picar.SunFounder_PCA9685 import Servo
import picar
from time import sleep
import cv2
import numpy as np
import picar
import os
import math

picar.setup()
db_file = "/home/pi/small_car/SunFounder_PiCar-V/remote_control/remote_control/driver/config"
fw = front_wheels.Front_Wheels(debug=False, db=db_file)
bw = back_wheels.Back_Wheels(debug=False, db=db_file)
#cam = camera.Camera(debug=False, db=db_file)
pan_servo = Servo.Servo(1)
tilt_servo = Servo.Servo(2)
pan_servo.write(90)
tilt_servo.write(90)
#cam.ready()
bw.ready()
fw.ready()
theta = 0
pan_angle = 90 # greater than 90 left ; less than 90 right
tilt_angle = 90 
    
pan_servo.write(pan_angle)
tilt_servo.write(tilt_angle)

fw_state = 'straight'
bw_state = 'brake'
SPEED = 0
minThreshold=150
maxThreshold=200
theta = 0
k_width = 640 
k_height = 480    
if __name__ == '__main__':
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret,frame = cap.read()
        if not ret:
            print("Cannot read video")
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edged = cv2.Canny(gray, 100, 200)
        lines = cv2.HoughLinesP(edged,1,np.pi/180,15,5,10)
        print(lines[0])
        for x in range(0, len(lines)):
            for x1,y1,x2,y2 in lines[x]:
                cv2.line(frame,(x1,y1),(x2,y2),(255,0,0),3)
                theta=theta+math.atan2((y2-y1),(x2-x1))
                print(theta)
        '''
        #bilateral = cv2.bilateralFilter(frame, 15, 75, 75)
        #img = cv2.cvtColor(bilateral, cv2.COLOR_BGR2GRAY)
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.imshow("origin", img)
        
        #blurred = cv2.GaussianBlur(img, (k_width, k_height), 0)
        
        _, thresh1 = cv2.threshold(img, 120, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        cv2.imshow('Otsu Threshold', thresh1)
        edges_img = cv2.Canny(thresh1, minThreshold, maxThreshold)
        #cv2.imshow("Canny", edges_img)
        # identify the area in the picture by pixel
        roi_pixel = np.array([[[10,470],[10,300],[620,300],[630,470]]])
        mask = np.zeros_like(edges_img) # the size of mask is same as the gary--> all 0
        mask = cv2.fillPoly(mask, roi_pixel, color = 255) # mask of ROI
        #cv2.imshow('mask', mask)
        img_mask = cv2.bitwise_and(edges_img,mask)
        cv2.imshow('img_mask', img_mask)
        lines = cv2.HoughLinesP(img_mask, 1, np.pi/ 180, 45,
                            minLineLength=10, maxLineGap=10)
        #print(lines[0])
        
        for x in range(0, len(lines)):
            for x1, x2, y1, y2 in lines[x]:
                cv2.line(img, (x1, y1), (x2, y2), (0, 255, 255), thickness=5)
                theta = theta + math.atan2((y2 - y1), (x2 - x1))
                #print(theta)
        cv2.imshow("output", img)
        '''
        threshold = 5
        if(theta>threshold):
            fw_state = 'left'
            bw_state = 'forward'   
            print("Go left")
        elif(theta<-threshold):
            fw_state = 'right'
            bw_state = 'forward'
            print("Go right")
        elif(abs(theta)<threshold):
            fw_state = 'straight'
            bw_state = 'forward'
            print("Go straight")
        else:
            fw_state = 'straight'
            bw_state = 'brake'
            print("Stop")
            
        if bw_state == 'brake':
            if fw_state == 'straight':
                bw.speed = 0
                bw.stop()
                fw.turn_straight()
            elif fw_state == 'left':
                bw.speed = 0
                bw.stop()
                fw.turn_left()
            elif fw_state == 'right':
                bw.speed = 0
                bw.stop()
                fw.turn_right()
        elif bw_state == 'forward':
            if fw_state == 'straight':
                bw.speed = SPEED
                bw.forward()
                fw.turn_straight()
            elif fw_state == 'left':
                bw.speed = SPEED
                bw.forward()
                fw.turn_left()
            elif fw_state == 'right':
                bw.speed = SPEED
                bw.forward()
                fw.turn_right()
        elif bw_state == 'backward':
            if fw_state == 'straight':
                bw.speed = SPEED
                bw.backward()
                fw.turn_straight()
            elif fw_state == 'left':
                bw.speed = SPEED
                bw.backward()
                fw.turn_left()
            elif fw_state == 'right':
                bw.speed = SPEED
                bw.backward()
                fw.turn_right()
        sleep(0.02)
        theta = 0
        if cv2.waitKey(1) == ord('q'):
            cv2,destroyAllWindows()
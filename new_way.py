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
bw.ready()
fw.ready()

threshold1 = 130
threshold2 = 220
theta=0

minLineLength = 5
maxLineGap = 10
k_width = 5
k_height = 5
max_slider = 10

# Linux System Serial Port
# ser = serial.Serial("/dev/ttyACM0", 115200, timeout=1)           # linux

# Read Image
cap = cv2.VideoCapture(0)
while True:
    ret,frame = cap.read()
# Resize width=500 height=300 incase of inputting raspi captured image
    #image = cv2.resize(frame,(r_width,r_height))
# Convert the image to gray-scale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
# given input image, kernel width =5 height = 5, Gaussian kernel standard deviation
    blurred = cv2.GaussianBlur(gray, (k_width, k_height), 0)
# Find the edges in the image using canny detector
    edged = cv2.Canny(blurred, threshold1, threshold2)
    
    roi_pixel = np.array([[[10,470],[10,280],[620,280],[630,470]]])
    mask = np.zeros_like(edged) # the size of mask is same as the gary--> all 0
    mask = cv2.fillPoly(mask, roi_pixel, color = 255) # mask of ROI
    #cv2.imshow('mask', mask)
    img_mask = cv2.bitwise_and(gray,mask)
    cv2.imshow("after mask", img_mask)
    
# Detect points that form a line
   
    lines = cv2.HoughLinesP(img_mask,1,np.pi/180,max_slider,minLineLength,maxLineGap)
    print(lines[0])
    for x in range(0, len(lines)):
        for x1,y1,x2,y2 in lines[x]:
            cv2.line(gray,(x1,y1),(x2,y2),(255,0,0),3)
            theta=theta+math.atan2((y2-y1),(x2-x1))
            print(theta)
    threshold=5
    if(theta>threshold):
        print("Go left")
    if(theta<-threshold):
        
        print("Go right")
    if(abs(theta)<threshold):
        print("Go straight")
    theta=0
    cv2.imshow("Gray Image",gray)
    #cv2.imshow("blurred",blurred)
    cv2.imshow("Edged",edged)
    #cv2.imshow("ROI",deal_img)  
    cv2.imshow("Line Detection",frame)
    if cv2.waitKey(1) == ord('q'):
        cv2.destroyAllWindows()
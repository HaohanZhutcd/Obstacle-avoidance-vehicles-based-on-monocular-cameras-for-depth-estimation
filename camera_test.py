from picar import front_wheels, back_wheels
from picar.SunFounder_PCA9685 import Servo
import picar
import time
import cv2
import numpy as np
import picar
import os

pan_servo = Servo.Servo(1)
tilt_servo = Servo.Servo(2)

pan_servo.write(90)
tilt_servo.write(90)

def main():
    pan_angle = 110 # greater than 90 left ; less than 90 right
    tilt_angle = 90 
    
    pan_servo.write(pan_angle)
    tilt_servo.write(tilt_angle)
if __name__ == '__main__':
    cap = cv2.VideoCapture(0)
    while True:
        _,frame = cap.read()
        cv2.imshow("1",frame)
        
        if cv2.waitKey(1) == ord("q"):
            cv2.destoryAllWindows()
            break
    
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
#======================================================
def capture_left_image():
    pan_angle = 95
    tilt_angle = 90
    pan_servo.write(pan_angle)
    tilt_servo.write(tilt_angle)
    cap = cv2.VideoCapture(0)

    n=1
    timeF = 10
    i=0
    while cap.isOpened():
        ret,frame = cap.read()
        if (n % timeF ==0):
            i+=1
            print(i)
            cv2.imwrite('/home/pi/car2/data/L/{}.jpg'.format(i),frame)
        n = n+1
        cv2.waitKey(1)
        if(n==11):
            break
    cap.release()
    cv2.destroyAllWindows()
#======================================================
def capture_right_image():
    pan_angle = 85
    tilt_angle = 90
    pan_servo.write(pan_angle)
    tilt_servo.write(tilt_angle)
    cap = cv2.VideoCapture(0)

    n=1
    timeF = 10
    i=0
    while cap.isOpened():
        ret,frame = cap.read()
        if (n % timeF ==0):
            i+=1
            print(i)
            cv2.imwrite('/home/pi/car2/data/R/{}.jpg'.format(i),frame)
        n = n+1
        cv2.waitKey(1)
        if(n==11):
            break
    cap.release()
    cv2.destroyAllWindows()    
if __name__ == "__main__":
    capture_right_image()
    time.sleep(1)
    capture_left_image()
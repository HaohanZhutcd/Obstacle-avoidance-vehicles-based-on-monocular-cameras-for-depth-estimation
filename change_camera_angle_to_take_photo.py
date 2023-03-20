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

'''
pan_servo.write(95)
tilt_servo.write(90)
'''
if __name__ == '__main__':
    pan_servo.write(90) # greater than 90 left ; less than 90 right
    tilt_servo.write(90) # >90 up, <90 down
    
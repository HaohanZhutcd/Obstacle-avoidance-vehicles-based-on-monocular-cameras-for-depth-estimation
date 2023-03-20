from picar import front_wheels, back_wheels
from picar.SunFounder_PCA9685 import Servo
import picar
from time import sleep
import cv2
import numpy as np
import picar
import os
#/home/pi/small_car/data/left.avi
filename = '/home/pi/small_car/data/left.avi'
picar.setup()
db_file = "/home/pi/small_car/SunFounder_PiCar-V/remote_control/remote_control/driver/config"
fw = front_wheels.Front_Wheels(debug=False, db=db_file)
bw = back_wheels.Back_Wheels(debug=False, db=db_file)
pan_servo = Servo.Servo(1)
tilt_servo = Servo.Servo(2)
bw.ready()
fw.ready()
class cameraComput(object):
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            self.cap.open()
    def gefFrame(self):
        ret,frame = self.cap.read()
        if ret:
            cv2.imshow("frame",frame)
            time.sleep(2)
        return frame
    
    def saveVideo(self,filepath,delays):
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        outputPath = filepath
        out = cv2.VideoWriter(outputPath,fourcc, 20.0, (640,480))
        startTime = time.time()
        while(self.cap.isOpened):
            ret,frame = self.cap.read()
            if ret:
                
                # frame = cv2.flip(frame,0)
                # write the flipped frame
                out.write(frame)
                cv2.imshow('frame',frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                break
            if time.time() - startTime > delays :
                break
        out.release()
        cv2.destroyAllWindows()
        return True
    
    def saveSnapshot(self,filepath):
        if self.cap.isOpened :
            ret,frame = self.cap.read()
            if ret:
                cv2.imwrite(filepath,frame)
            else:
                print("save snapshot fail")
                return False
        return True
 
    def releaseDevice(self):
        self.cap.release()
 
 
    def reOpen(self):
        if not self.cap.isOpened():
            print("re opened device")
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                self.cap.open()
def test():
    import  time
    cap = cv2.VideoCapture(0)
    while True:
        ret,photo=cap.read()
        cv2.imshow('Please Take Your Photo!!',photo)
        key=cv2.waitKey(2)
        if key==ord(" "):
            filename = time.strftime('%Y%m%d-%H%M%S') + ".jpg"
            cv2.imwrite(filename,photo) 
        if key==ord("q"):
            cap.release()
            break
            pass
test()

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

#==========================
#       ▄▀▄     ▄▀▄       
#      ▄█░░▀▀▀▀▀░░█▄       
#  ▄▄  █░░░░░░░░░░░█  ▄▄  
# █▄▄█ █░░▀░░┬░░▀░░█ █▄▄█
#=========================
def left_image(left_detection):
    if left_detection == True:
        pan_angle = 95
        tilt_angle = 90
        pan_servo.write(pan_angle)
        tilt_servo.write(tilt_angle)
        
        capture_duration = 10
        cap = cv2.VideoCapture(0)
        left_fourcc = cv2.VideoWriter_fourcc(*'XVID')
        left_out = cv2.VideoWriter('/home/pi/small_car/data/left.avi', left_fourcc, 30.0, (640,480))
        
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
#==========================
#       ▄▀▄     ▄▀▄       
#      ▄█░░▀▀▀▀▀░░█▄       
#  ▄▄  █░░░░░░░░░░░█  ▄▄  
# █▄▄█ █░░▀░░┬░░▀░░█ █▄▄█
#=========================
def right_image(right_detection):
    if right_detection == True:
        pan_angle = 85
        tilt_angle = 90
        pan_servo.write(pan_angle)
        tilt_servo.write(tilt_angle)
        
        capture_duration = 10
        cap = cv2.VideoCapture(0)
        right_fourcc = cv2.VideoWriter_fourcc(*'XVID')
        right_out = cv2.VideoWriter('/home/pi/small_car/data/right.avi', right_fourcc, 30.0, (640,480))
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
#==========================
#       ▄▀▄     ▄▀▄       
#      ▄█░░▀▀▀▀▀░░█▄       
#  ▄▄  █░░░░░░░░░░░█  ▄▄  
# █▄▄█ █░░▀░░┬░░▀░░█ █▄▄█
#=========================
def capture_images():
    pan_angle = 90 # greater than 90 left ; less than 90 right
    tilt_angle = 90 
    pan_servo.write(pan_angle)
    tilt_servo.write(tilt_angle)
    time.sleep(2)
    
    
    start = time.time()
    T = 10
    count = 0
    
    cap_L = cv2.VideoCapture("/home/pi/small_car/data/left.avi")
    cap_R = cv2.VideoCapture("/home/pi/small_car/data/right.avi")
    output_path = "/home/pi/small_car/data/"
    while True:
        #timer = T - int(time.time() - start)
        retL,frame_L = cap_L.read()
        retR,frame_R = cap_R.read()
        if retL == True and retR == True:
            cv2.imshow("L",frame_L)
            cv2.imshow("R",frame_R)
            grayL = cv2.cvtColor(frame_L,cv2.COLOR_BGR2GRAY)
            grayR = cv2.cvtColor(frame_R,cv2.COLOR_BGR2GRAY)
            
            retR, cornersR = cv2.findChessboardCorners(grayR,(5,7),None)
            retL, cornersL = cv2.findChessboardCorners(grayL,(5,7),None)
            if (retR == True) and (retL == True): #and timer <=0:
                count+=1
                cv2.imwrite(output_path+'R1/img%d.png'%count,frame_R)
                cv2.imwrite(output_path+'L1/img%d.png'%count,frame_L)
            #if timer <=0:
            #    start = time.time()
            if cv2.waitKey(1) == ord("q"):
                print("close camera")
                break
        else:
            print("not ret in main")
            break
    cap_L.release()
    cap_R.release()
    cv2.destroyAllWindows()
if __name__ == "__main__":
    #while(1):
    left_detection = True
    left_image(left_detection)
    #time.sleep(10)        
    left_detection = False
    left_image(left_detection)
            
    right_detection = True
    right_image(right_detection)
    #time.sleep(10)       
    right_detection = False
    right_image(right_detection)    
    capture_images()
'''
def left_image(left_detection):
    if left_detection == True:
        pan_angle = 95
        tilt_angle = 90
        pan_servo.write(pan_angle)
        tilt_servo.write(tilt_angle)
        
        capture_duration = 10
        cap = cv2.VideoCapture(0)
        left_fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        left_out = cv2.VideoWriter('/home/pi/small_car/videodata/left.avi', left_fourcc, 30.0, (640,480))
        
        start_time = time.time()
        while(int(time.time() - start_time) < capture_duration):
            ret,frame_left = cap.read()
            if not ret:
                print("Not ret")
                break
            left_out.write(frame_left)
            cv2.imshow("left_orig",frame_left)
            cv2.waitKey(1)
            #if cv2.waitKey(1) & 0xFF == ord("q"):
            #    break
        cap.release()
        left_out.release()
        cv2.destroyAllWindows()
    if left_detection == False:
        print("not left turn")
        return

def right_image(right_detection):
    if right_detection == True:
        pan_angle = 85
        tilt_angle = 90
        pan_servo.write(pan_angle)
        tilt_servo.write(tilt_angle)
        
        capture_duration = 10
        cap = cv2.VideoCapture(0)
        right_fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        right_out = cv2.VideoWriter('/home/pi/small_car/videodata/right.avi', right_fourcc, 30.0, (640,480))
        start_time = time.time()
        
        while(int(time.time() - start_time) < capture_duration):
            ret,frame_right = cap.read()
            if not ret:
                print("Not ret")
                break
            right_out.write(frame_right)
            cv2.imshow("right_orig",frame_right)
            cv2.waitKey(1)
            #if cv2.waitKey(1) & 0xFF == ord("q"):
            #    break
        cap.release()
        right_out.release()
        cv2.destroyAllWindows()
    if right_detection == False:
        print("not right turn")
        return
if __name__ == '__main__':
    pan_angle = 90 # greater than 90 left ; less than 90 right
    tilt_angle = 90 
    pan_servo.write(pan_angle)
    tilt_servo.write(tilt_angle)
    time.sleep(2)
    while(1):
        left_detection = True
        left_image(left_detection)
        
        #left_detection = False
        #left_image(left_detection)
        
        right_detection = True
        right_image(right_detection)
        
        #right_detection = False
        #right_image(right_detection)
        cap_L = cv2.VideoCapture("./videodata/left.avi")
        cap_R = cv2.VideoCapture("./videodata/right.avi")
        
        while True:
            retL,frame_L = cap_L.read()
            retR,frame_R = cap_R.read()
            cv2.imshow("L",frame_L)
            cv2.imshow("R",frame_R)
            
            if cv2.waitKey(1) == ord("q"):
                break
        cap_L.release()
        cap_R.release()
        cv2.destroyAllWindows()
'''        

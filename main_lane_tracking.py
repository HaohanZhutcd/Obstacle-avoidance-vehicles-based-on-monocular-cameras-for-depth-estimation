#      ▄▀▄     ▄▀▄
#     ▄█░░▀▀▀▀▀░░█▄
# ▄▄  █░░░░░░░░░░░█  ▄▄
#█▄▄█ █░░▀░░┬░░▀░░█ █▄▄█

from picar import front_wheels, back_wheels
from picar.SunFounder_PCA9685 import Servo
import picar
from time import sleep
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

if __name__ == '__main__':
    num_lane_point = 4 # detect the particular point in a area
    SPEED = 0
    pan_angle = 90 # greater than 90 left ; less than 90 right
    tilt_angle = 90 
    pan_servo.write(pan_angle)
    tilt_servo.write(tilt_angle)
    bw_state = 'forward'
    fw_state = 'straight'
    cap = cv2.VideoCapture(0)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("no Video input")
        original_frame = np.copy(frame)
        
        img = cv2.blur(original_frame,(5,5))
        #cv2.imshow('blur', img)
        
        _, _, red_img = cv2.split(img)
        #cv2.imshow("red_channel", red_img)
        
        _, dst = cv2.threshold(red_img, 120, 255, cv2.THRESH_BINARY)
        #cv2.imshow("gray",dst)
        #print(str(dst.shape))
        height, width = dst.shape # [480,640]
        half_width = int (width/2) # 320
        
        right_line_pos = np.zeros((num_lane_point, 1)) # num_lane_point = 4
        left_line_pos = np.zeros((num_lane_point, 1))
        
        img_out = cv2.cvtColor(dst, cv2.COLOR_GRAY2RGB)
        cv2.imshow('img_out', img_out)
        
        for i in range(num_lane_point):   # each detected point on the lane
            detect_height = height - 20 * (i+1)
            print("detect_height:",detect_height)
            detect_area_left = dst[detect_height, 0: half_width - 1]  # divide the image into two parts: left and right (this may cause problems, which can be optimized in the future)
            detect_area_right = dst[detect_height, half_width: width-1]
        line_left = np.where(detect_area_left == 0)   # extract  zero pixels' index
        line_right = np.where(detect_area_right == 0)
        if len(line_left[0]):
            left_line_pos[i] = int(np.max(line_left))  # set the most internal pixel as the lane point
        else:
            left_line_pos[i] = 0  # if haven't detected any zero pixel, set the lane point as 0

        if len(line_right[0]):
            right_line_pos[i] = int(np.min(line_right))
        else:
            right_line_pos[i] = half_width - 1               

        if left_line_pos[i] != 0:   # draw the lane points on the binary image
            img_out = cv2.circle(img_out, (int(left_line_pos[i]), int(detect_height)),
                                 4, (0, 0, 255), thickness=10)
        if right_line_pos[i] != half_width - 1:
            img_out = cv2.circle(img_out, (int(half_width + right_line_pos[i]), int(detect_height)),
                                 4, (0, 0, 255), thickness=10)
        cv2.imshow("output",img_out)
        left_max = np.max(left_line_pos)
        right_min = np.max(right_line_pos)
        print(left_max)
        print(right_min)
        #print(half_width)
            
        if left_max == 0 and right_min == half_width - 1:
                pass
        elif left_max == 0: 
            if right_min > half_width - 100:
                fw_state = 'Straight'
                bw_state = 'Forward'
                print("forward and turn straight")
            elif right_min < 100:
                fw_state = 'Left'
                bw_state = 'Brake'
                print("stop and turn left")
            else:   
                fw_state = 'Left'
                bw_state = 'Forward'
                print("forward and turn left")
        elif right_min == half_width - 1:
            if left_max <100:
                fw_state = 'Straight'
                bw_state = 'Forward'
                print("forward and turn straight")
            elif left_max > half_width - 100:
                fw_state = 'Right'
                bw_state = 'Brake'
                print("stop and turn right")
            else:
                fw_state = 'Right'
                bw_state = 'Forward'
                print("forward and turn right")
        else:
            fw_state = 'Straight'
            bw_state = 'Forward'
            print("Go straight!")
        '''           
############################ motion control #####################################
        if bw_state == 'Brake':
            if fw_state == 'Left':
                bw.speed = SPEED - SPEED
                bw.stop()
                fw.turn_left()
            elif fw_state == 'Right':
                bw.speed = SPEED - SPEED
                bw.stop()
                fw.turn_right()
            elif fw_state == 'Straight':
                bw.speed = SPEED - SPEED
                bw.stop()
                fw.turn_straight()
        elif bw_state == 'Forward':
            if fw_state == 'Left':
                bw.speed = SPEED
                bw.forward()
                fw.turn_left()
            elif fw_state == 'Right':
                bw.speed = SPEED
                bw.forward()
                fw.turn_right()
            elif fw_state == 'Straight':
                bw.speed = SPEED
                bw.forward()
                fw.turn_straight()
        elif bw_state == 'Backward':
            if fw_state == 'Left':
                bw.speed = SPEED
                bw.backward()
                fw.turn_left()
            elif fw_state == 'Right':
                bw.speed = SPEED
                bw.backward()
                fw.turn_right()
            elif fw_state == 'Brake':
                bw.speed = SPEED
                bw.backward()
                fw.turn_straight()
        '''
        if cv2.waitKey(1) == ord('q'):
            cv2.destroyAllWindows()
            break
    
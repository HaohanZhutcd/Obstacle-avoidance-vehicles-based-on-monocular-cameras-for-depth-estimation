def detection_F(detection):
    if detection == True:
        environment_state = "safe"
        pan_angle = 90 # greater than 90 left ; less than 90 right
        tilt_angle = 90 
        pan_servo.write(pan_angle)
        tilt_servo.write(tilt_angle)
        
        num_lane_point = 5
        cap = cv2.VideoCapture(0)
        start_time = time.time()
        capture_duration = 5
        while cap.isOpened():
            ret,frame = cap.read()
            if not ret:
                print("Function lane_following cannot read the ret")
                break
            original_frame = np.copy(frame)
            cv2.imshow("original_frame",original_frame)
            img = cv2.blur(original_frame,(5,5))
            _, _, red_img = cv2.split(img)
            _, dst = cv2.threshold(red_img, 120, 255, cv2.THRESH_BINARY)
            height, width = dst.shape # [480,640]
            half_width = int (width/2) # 320
            right_line_pos = np.zeros((num_lane_point, 1)) # num_lane_point = 4
            left_line_pos = np.zeros((num_lane_point, 1))
            img_out = cv2.cvtColor(dst, cv2.COLOR_GRAY2RGB)
            cv2.imshow('img_out', img_out)
            
            for i in range(num_lane_point):   # each detected point on the lane
                detect_height = 440 - 20 * (i+1)
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
            #print("left dist:",left_max)
            #print("right dist:",half_width + right_min)
            #print(half_width)
            
            if left_max == 0 and right_min == half_width - 1:
                environment_state ='danger'
            elif half_width+right_min - left_max < half_width-2 :
                environment_state ='danger'
            else:
                environment_state = 'safe'
                
            if environment_state == 'danger':
                print("There is an obstacle in the safe distance")
                bw.stop()
                break
            else:
                print("safe")
                pass
            cv2.waitKey(1)
            if time.time() - start_time > capture_duration:
                break
        cap.release()
        cv2.destroyAllWindows()
        return environment_state
    else:
        pass
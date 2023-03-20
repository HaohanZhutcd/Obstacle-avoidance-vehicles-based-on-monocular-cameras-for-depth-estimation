#from picar import front_wheels, back_wheels
#from picar.SunFounder_PCA9685 import Servo
#import picar
#from time import sleep
import cv2
import numpy as np
#import os
#import picamera

#picar.setup()
#db_file = "/home/pi/SunFounder_PiCar-V/remote_control/remote_control/driver/config"
#fw = front_wheels.Front_Wheels(debug=False, db=db_file)
#bw = back_wheels.Back_Wheels(debug=False, db=db_file)
#cam = camera.Camera(debug=False, db=db_file)
#cam.ready()
#bw.ready()
#fw.ready()
#SPEED = 60
#bw_status = 0
#back wheels code
#bw.speed()
#bw.forward()
#bw.backward()
#bw.stop()
#front wheels code
#fw.turn_left()
#fw.turn_right()
#fw.turn_straight()
#time delay
#sleep(time)

# create a trackbar windows to adjust Canny threshold
cv2.namedWindow('edge threshold')
cv2.createTrackbar('minThreshold','edge threshold',400,1000,lambda x:x)
cv2.createTrackbar('maxThreshold','edge threshold',1000,1000,lambda x:x)

def canny_edge_img(color_img):
    #cap = cv2.VideoCapture(0)
    #ret, frame = cap.read()
    
    gray = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)
    #cv2.imshow("gray",gray)
    #minThreshold = 50, maxThreshold = 100
    minThreshold = cv2.getTrackbarPos('minThreshold','edge threshold')
    maxThreshold = cv2.getTrackbarPos('maxThreshold','edge threshold')
    edges_img = cv2.Canny(gray, minThreshold, maxThreshold)
    
    #cv2.imshow("Canny edges detection", edges_img)
    
    return edges_img
def region_of_interest(gray):
    # identify the area in the picture by pixel
    roi_pixel = np.array([[[0,480],[220,320],[580,320],[640,480]]])
    mask = np.zeros_like(gray) # the size of mask is same as the gary--> all 0
    mask = cv2.fillPoly(mask, roi_pixel, color = 255) # mask of ROI
    cv2.imshow('mask', mask)
    img_mask = cv2.bitwise_and(gray,mask)
    cv2.imshow('img_mask', img_mask)
    return img_mask

def get_straightcarLines(edge_img):
    
    def calculate_slope(line):
        # calculate the slope of straight lines
        # the reason is to use Hough algorithm to fit the left and right lines
        x_1,y_1,x_2,y_2 = line[0]
        return (y_2 - y_1) / (x_2 - x_1)
    
    
    def reject_different_slope_lines(lines, threshold=0.12):
        # list all lines detected
        slopes = [calculate_slope(line) for line in lines]
        while len(lines) > 0:
            mean = np.mean(slopes) # Slope average of all line segments slopes
            # Calculate the difference between all line segments and the mean slope
            diff = [abs(s - mean) for s in slopes] 
            idx = np.argmax(diff) # Returns the indices of the maximum values along an axis
            if diff[idx] > threshold:
                slopes.pop(idx) # pop the lines
                lines.pop(idx)  # pop the lines
            else:
                break
        return lines
    
    
    def least_squares_fit(lines):
        # Pull high-dimensional array to one dimension
        # [[1,2],[3,4]] -> [1,2,3,4]
        # 1.Calculate all coordinate points
        x_coords = np.ravel([[line[0][0], line[0][2]] for line in lines])
        y_coords = np.ravel([[line[0][1], line[0][3]] for line in lines])
        # polynomial fit, [k, b], y = kx+b, degree = 1
        # 2.fit the staright line, get polynomial coefficients
        poly = np.polyfit(x_coords, y_coords, deg=1)
        # polynomial evaluation
        # 3. Identify unique line
        point_min = (np.min(x_coords), np.polyval(poly, np.min(x_coords)))
        point_max = (np.max(x_coords), np.polyval(poly, np.max(x_coords)))
        return np.array([point_min, point_max], dtype=int)
    
    # Hough Transform
    lines = cv2.HoughLinesP(edge_img, 1, np.pi / 180, 15, minLineLength=40, maxLineGap=20)
    # make a classification
    left_lines = [line for line in lines if calculate_slope(line) > 0]
    right_lines = [line for line in lines if calculate_slope(line) < 0]
    #left_lines = []
    #for line in lines:
    #    if calculate_slope(line) > 0:
    #        left_lines.append(line)
    #right_lines = []
    #for line in lines:
    #    if calculate_slope(line) < 0:
    #        right_lines.append(line)
    
    # test the lines which are detected
    #print("before filter:")
    #print("lines:" + str(len(lines)))
    #print("left lines:" + str(len(left_lines)))
    #print("right lines:" + str(len(right_lines))) 
    
    # Outlier filtering
    
    left_lines = reject_different_slope_lines(left_lines)
    right_lines = reject_different_slope_lines(right_lines)

    #print("after filter:")
    #print("lines:" + str(len(lines)))
    #print("left lines:" + str(len(left_lines)))
    #print("right lines:" + str(len(right_lines)))
    
    #print("left lane:" + str(least_squares_fit(left_lines)))
    #print("right lane:" + str(least_squares_fit(right_lines)))
    return least_squares_fit(left_lines), least_squares_fit(right_lines)

def draw_lines(img, lines):
    left_line, right_line = lines
    cv2.line(img, tuple(left_line[0]), tuple(left_line[1]), color=(0, 255, 255),
             thickness=5)
    cv2.line(img, tuple(right_line[0]), tuple(right_line[1]),
             color=(0, 255, 255), thickness=5)
    #cv2.imshow("lines" , img)

def show_lane(color_img):
    edge_img = canny_edge_img(color_img)
    mask_gray_img = region_of_interest(edge_img)
    straightlines = get_straightcarLines(mask_gray_img)
    draw_lines(color_img, straightlines)
    return color_img

def lane_following_video_process():
    cap = cv2.VideoCapture(0)
    #fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    #outfile = cv2.VideoWriter('output.avi', fourcc, 25., (1280, 368))
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Cannot recceive frame (stream end?). Exiting..")
        origin = np.copy(frame)
        frame = show_lane(frame)
        output = np.concatenate((origin, frame), axis=1)
        #outfile.write(output)
        cv2.imshow("output", output)
        if cv2.waitKey(1) == ord('q'):
            cv2.destroyAllWindows()
            break
        
if __name__ == '__main__':
    lane_following_video_process()
    '''
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Cannot recceive frame (stream end?). Exiting..")
        edge_img = canny_edge_img(frame)
        mask_gray_img = region_of_interest(edge_img)
        straightlines = get_straightcarLines(mask_gray_img)
        draw_lines(frame, straightlines)
        if cv2.waitKey(1) == ord('q'):
            cv2.destroyAllWindows()
            break
    '''
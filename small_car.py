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
bw.ready()
fw.ready()
SPEED = 50
'''
bw.ready()
bw.speed = 40
bw.forward()
sleep(1)
bw.stop()
'''
def get_Otsuimg(origin_img, minThreshold = 130, maxThreshold = 220):
    #bilateral = cv2.bilateralFilter(frame, 15, 75, 75)
    #img = cv2.cvtColor(bilateral, cv2.COLOR_BGR2GRAY)
    img = cv2.cvtColor(origin_img, cv2.COLOR_BGR2GRAY)
    _, thresh1 = cv2.threshold(img, 120, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cv2.imshow('Otsu Threshold', thresh1)
    #cv2.namedWindow('edge threshold')
    #cv2.createTrackbar('minThreshold','edge threshold',75,1000,lambda x:x)
    #cv2.createTrackbar('maxThreshold','edge threshold',150,1000,lambda x:x)
    #minThreshold = cv2.getTrackbarPos('minThreshold','edge threshold')
    #maxThreshold = cv2.getTrackbarPos('maxThreshold','edge threshold')
    edges_img = cv2.Canny(thresh1, minThreshold, maxThreshold)
    cv2.imshow('edge',edges_img)
    return edges_img
    '''
    initial_img = cv2.VideoCapture(0)
    while initial_img.isOpened():
        ret, frame = initial_img.read()
        if not ret:
            print("Cannot read the stream")
        #cv2.GaussinBlur(frame,(gaussian_ksize, gaussian_ksize),gaussian_sigmax)
        
        # Apply bilateral filter with d = 15,
        # sigmaColor = sigmaSpace = 75.
        bilateral = cv2.bilateralFilter(frame, 15, 75, 75)
        img = cv2.cvtColor(bilateral, cv2.COLOR_BGR2GRAY)
        #img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, thresh1 = cv2.threshold(img, 120, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)     
        cv2.imshow('Otsu Threshold', thresh1)
        if cv2.waitKey(1)  == ord('Q'):
            cv2.destroyAllWindows()
            break
    '''
def region_of_interest(thresh1):
    # identify the area in the picture by pixel
    roi_pixel = np.array([[[10,470],[150,300],[620,300],[630,470]]])
    mask = np.zeros_like(thresh1) # the size of mask is same as the gary--> all 0
    mask = cv2.fillPoly(mask, roi_pixel, color = 255) # mask of ROI
    cv2.imshow('mask', mask)
    img_mask = cv2.bitwise_and(thresh1,mask)
    cv2.imshow('img_mask', img_mask)
    return img_mask
def get_straightcarLines(mask_gray_img):
    left_lane = []
    right_lane = []
    def calculate_slope(line):
        x_1,y_1,x_2,y_2 = line[0]
        line_slope = (y_2 - y_1) / (x_2 - x_1)
        return line_slope
    def reject_different_slope_lines(lines, threshold=0.3):
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
        x_coords = np.ravel([[line[0][0], line[0][2]] for line in lines])
        y_coords = np.ravel([[line[0][1], line[0][3]] for line in lines])
        poly = np.polyfit(x_coords, y_coords, deg=2)
        point_min = (np.min(x_coords), np.polyval(poly, np.min(x_coords)))
        point_max = (np.max(x_coords), np.polyval(poly, np.max(x_coords)))
        return np.array([point_min, point_max], dtype=int)
    lines = cv2.HoughLinesP(mask_gray_img, 1, np.pi / 180, 15, minLineLength=15, maxLineGap=10)
    
    left_lines = [line for line in lines if calculate_slope(line) > 0]
    right_lines = [line for line in lines if calculate_slope(line) < 0]
    left_lines = reject_different_slope_lines(left_lines)
    right_lines = reject_different_slope_lines(right_lines)
    
    return least_squares_fit(left_lines), least_squares_fit(right_lines)
'''
def get_straightcarLines(mask_gray_img):
    
    def calculate_slope(line):
        # calculate the slope of straight lines
        # the reason is to use Hough algorithm to fit the left and right lines
        x_1,y_1,x_2,y_2 = line[0]
        return (y_2 - y_1) / (x_2 - x_1)
    
    
    def reject_different_slope_lines(lines, threshold=0.2):
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
    lines = cv2.HoughLinesP(mask_gray_img, 1, np.pi / 180, 25, minLineLength=10, maxLineGap=20)
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
'''
def draw_lines(img, lines):
    left_line, right_line = lines
    cv2.line(img, tuple(left_line[0]), tuple(left_line[1]), color=(0, 255, 255),
             thickness=5)
    cv2.line(img, tuple(right_line[0]), tuple(right_line[1]),
             color=(0, 255, 255), thickness=5)
    #cv2.imshow("lines" , img)

def show_lane(origin_img):
    edge_img = get_Otsuimg(origin_img)
    mask_gray_img = region_of_interest(edge_img)
    straightlines = get_straightcarLines(mask_gray_img)
    draw_lines(origin_img, straightlines)
    return origin_img
'''
class CarDetect(object):
    def __init__(self):
        self.track_length = 20
        self.near_pos = 400
        self.medium_pos = 300
        self.far_pos = 200
        pass

    def group_consecutives(self, vals, step=1):
        run = []
        result = []
        expect = None
        for v in vals[0]:
            if expect is None or abs(v - expect) < 2:
                run.append(v)
            else:
                result.append(run)
                run = []
            expect = v + step
        if result != []:
            print(len(result[0]))
        return result
        # return [vals[0]]

    def denoise(self, areas, pos):
        result = []
        for a in areas:
            if pos == 'near' and abs(len(a) - 125) < 15:
                result.append(a)
            elif pos == 'medium' and abs(len(a) - 90) < 15:
                result.append(a)
            elif pos == 'far' and abs(len(a) - 75) < 15:
                result.append(a)
        print(result)
        return result
        # return areas

    def get_center(self, line):
        if len(line) == 0:
            center = -1
        elif len(line) == 1:
            count = len(line[0])
            index = line
            center = (index[0][count - 1] + index[0][0]) / 2
        else:
            print(len(line))
            center = -1
        return center

    def get_center(self, line):
        index = np.where(line == 255)
        if index == 0:
            index = 1
        sum = np.sum(line == 255)
        if sum > 0:
            center = (index[0][sum - 1] + index[0][0]) / 2
        else:
            center = -1
        return center

    def LineTrack(self, img, VideoReturn):  # 检测黑线（利用大津法进行二值化）
        x_bias = 0  # initialize the bias

        img = cv2.blur(img, (5, 5))  # denoising
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)  # rgb to gray

        _, dst = cv2.threshold(gray_img, 70, 255, cv2.THRESH_BINARY)  # OTSU binaryzation
        dst = cv2.dilate(dst, None, iterations=2)  # dilate image to add the white area
        # dst = cv2.erode(dst, None, iterations=6)

        near_line = self.group_consecutives(np.where(dst[self.near_pos] == 0))
        medium_line = self.group_consecutives(np.where(dst[self.medium_pos] == 0))
        far_line = self.group_consecutives(np.where(dst[self.far_pos] == 0))

        near_line = self.denoise(near_line, 'near')
        medium_line = self.denoise(medium_line, 'medium')
        far_line = self.denoise(far_line, 'far')

        near_center = self.get_center(near_line)
        medium_center = self.get_center(medium_line)
        far_center = self.get_center(far_line)

        if VideoReturn:  # if it needs to return the frame with the detected tennis
            img_out = copy.copy(dst)
            img_out = cv2.circle(img_out, (int(near_center), self.near_pos), 4, (0, 0, 255), thickness=10)
            img_out = cv2.circle(img_out, (int(medium_center), self.medium_pos), 4, (0, 0, 255), thickness=10)
            img_out = cv2.circle(img_out, (int(far_center), self.far_pos), 4, (0, 0, 255), thickness=10)
            return img_out, near_center, medium_center, far_center
        else:  # if it only needs to return the position of the detected tennis
            return near_center, medium_center, far_center

class Car(CarDetect):  # create class Car, which derives all the modules
    def __init__(self):
        CarDetect.__init__(self)

def lane_detection():
    car = Car()
    VideoReturn = True
    num_lane_point = 5
    ForB = 'Forward'
    LorR = 'Brake'
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        camera,rawCapture=cap.read()
        cv2.imshow("origin", rawCapture)
    #camera, rawCapture = car.CameraInit()  # Initialize the PiCamera
    for raw_frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
        frame_origin = raw_frame.array
        img = cv2.blur(frame_origin, (5, 5))  # denoising
    #cap = cv2.VideoCapture(0)
    #while cap.isOpened():
        #ret,frame = cap.read()
        #if not ret:
        #    print("cannot read the stream.")
        #img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #img = cv2.blur(frame, (5, 5))
        _, _, red_img = cv2.split(img)
        cv2.imshow("red",red_img)
        #_, dst = cv2.threshold(frame, 120, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        _, dst = cv2.threshold(red_img, 125, 255, cv2.THRESH_BINARY)  # binaryzation, the thresold deponds on the light in the environment
        height,width = dst.shape
        half_width = int (width/2)
        right_line_pos = np.zeros((num_lane_point, 1))
        left_line_pos = np.zeros((num_lane_point, 1))
        img_out = cv2.cvtColor(dst, cv2.COLOR_GRAY2RGB)
        cv2.imshow("1",dst)
        for i in range(num_lane_point):   # each detected point on the lane
            detect_height = height - 15 * (i+1)
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
                img_out = cv2.circle(img_out, (left_line_pos[i], detect_height), 4, (0, 0, 255), thickness=10)
            if right_line_pos[i] != half_width - 1:
                img_out = cv2.circle(img_out, (half_width + right_line_pos[i], detect_height), 4, (0, 0, 255), thickness=10)
        if VideoReturn:  # detect the tennis & transmit the frames to PC
                car.VideoTransmission(img_out)
        left_max = np.max(left_line_pos)
        right_min = np.min(right_line_pos)  # choose the most internal lane point for decision making
        if left_max == 0 and right_min == half_width - 1: 
            pass
        elif left_max == 0: 
            if right_min > half_width - 100:  
                ForB = 'Forward'
                LorR = 'Brake'
            elif right_min < 100:   
                ForB = 'Brake'
                LorR = 'Left'
            else:   
                ForB = 'Forward'
                LorR = 'Left'
        elif right_min == half_width - 1:
            if left_max <100:
                ForB = 'Forward'
                LorR = 'Brake'
            elif left_max > half_width - 100:
                ForB = 'Brake'
                LorR = 'Right'
            else:
                ForB = 'Forward'
                LorR = 'Right'
        else:
            ForB = 'Forward'
            LorR = 'Brake'
            
        if ForB == 'Brake':
            if LorR == 'Left':
                #car.left(turn_left_speed)
                SPEED = 0
                bw.stop()
                fw.turn_left()
            elif LorR == 'Right':
                #car.right(turn_right_speed)
                SPEED = 0
                bw.stop()
                fw.turn_right()
            elif LorR == 'Brake':
                SPEED = 0
                #car.brake()
                bw.stop()
                fw.turn_straight()
        elif ForB == 'Forward':
            if LorR == 'Left':
                #car.forward_turn(speed_low, speed_high)
                SPEED = 60
                fw.turn_left()
                bw.forward()
            elif LorR == 'Right':
                #car.forward_turn(speed_high, speed_low)
                SPEED = 60
                fw.turn_right()
                bw.forward()
            elif LorR == 'Brake':
                #car.forward(forward_speed)
                SPEED = 60
                fw.turn_straight()
                bw.forward()
        elif ForB == 'Backward':
            if LorR == 'Left':
                #car.left(turn_left_speed)
                SPEED = 60
                fw.turn_left()
                bw.backward()
            elif LorR == 'Right':
                #car.right(turn_right_speed)
                SPEED = 60
                fw.turn_right()
                bw.backward()
            elif LorR == 'Brake':
                #car.back(40)
                SPEED = 60
                fw.turn_straight()
                bw.backward()
                
'''           
if __name__ == '__main__':
    '''
    lane_detection()
    '''
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret,frame = cap.read()
        if not ret:
            print("cannot read the stream.")
        #edge_img = get_Otsuimg(frame)
        #mask_gray_img = region_of_interest(edge_img)
        #straightlines = get_straightcarLines(mask_gray_img)
        #cv2.imshow("edge_img",edge_img)
        output = show_lane(frame)
        cv2.imshow('output', output)
        if cv2.waitKey(1) == ord('q'):
            cv2.destroyAllWindows()
            break
    

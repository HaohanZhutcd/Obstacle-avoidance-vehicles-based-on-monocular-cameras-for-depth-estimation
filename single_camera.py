import numpy as np
import cv2
 
def find_marker(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  
    gray = cv2.GaussianBlur(gray, (5, 5), 1)
    
    #cv2.imshow("gray",gray)
    edged = cv2.Canny(gray, 30, 130)
    cv2.imshow("edge",edged)
    (cnts, _) = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)  
    c = max(cnts, key = cv2.contourArea)
    return cv2.minAreaRect(c)
 
def distance_to_camera(knownWidth, focalLength, perWidth):  
    return (knownWidth * focalLength) / perWidth            
 

KNOWN_DISTANCE = 11.02362

KNOWN_WIDTH = 13.385827
KNOWN_HEIGHT = 7.480314

IMAGE_PATHS = ["/home/pi/car2/f/image1.jpg"]
 

image = cv2.imread(IMAGE_PATHS[0])  
marker = find_marker(image)           
focalLength = (marker[1][0] * KNOWN_DISTANCE) / KNOWN_WIDTH  
print('focalLength = ',focalLength)
camera = cv2.VideoCapture(0)



for imagePath in IMAGE_PATHS:
    image = cv2.imread(imagePath)
    marker = find_marker(image)
    inches = distance_to_camera(KNOWN_WIDTH, focalLength, marker[1][0])

    box = np.int0(cv2.boxPoints(marker))
    cv2.drawContours(image, [box], -1, (0, 255, 0), 2)
    cv2.putText(image, "%.2fcm" % (inches*30.48/12),
        (image.shape[1] - 200, image.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX,
        2.0, (0, 255, 0), 3)
    cv2.imshow("image", image)
    #if cv2.waitKey(10)>=0:
        #	break
    cv2.waitKey(0)
while camera.isOpened():
    (grabbed, frame) = camera.read()
    if not grabbed:
        break

    marker = find_marker(frame)
    if marker == 0:
        print(marker)
        continue
    inches = distance_to_camera(KNOWN_WIDTH, focalLength, marker[1][0])

    # draw a bounding box around the image and display it
    box = np.int0(cv2.boxPoints(marker))
    cv2.drawContours(frame, [box], -1, (0, 255, 0), 2)
    cv2.putText(frame, "%.2fcm" % (inches *30.48/ 12),
                (frame.shape[1] - 200, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX,
                2.0, (0, 255, 0), 3)

    cv2.imshow("capture", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
camera.release()
cv2.destroyAllWindows() 


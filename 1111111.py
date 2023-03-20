import cv2
import numpy as np
import camera_configs

cv2.namedWindow('depth')
cv2.moveWindow('left',0,0)
cv2.moveWindow('right',640,0)
cv2.createTrackbar('num','depth',8,15, lambda x: None)
cv2.createTrackbar('blockSize','depth',8,255,lambda x: None)

def callbackFunc(e,x,y,f,p):
    if e == cv2.EVENT_LBUTTONDOWN:
        print (threeD[y][x])

cv2.setMouseCallback('depth',callbackFunc,None)


capL = cv2.VideoCapture("/home/pi/car2/data/video/L/left.avi")
capR = cv2.VideoCapture("/home/pi/car2/data/video/R/right.avi")

while True:
    ret,frame1 = capL.read()

    
    
    ret,frame2 = capR.read()

    img1_rectified = cv2.remap(frame1, camera_configs.left_map1, camera_configs.left_map2, cv2.INTER_LINEAR)
    img2_rectified = cv2.remap(frame2, camera_configs.right_map1, camera_configs.right_map2, cv2.INTER_LINEAR)

    imgL = cv2.cvtColor(img1_rectified, cv2.COLOR_BGR2GRAY)
    imgR = cv2.cvtColor(img2_rectified, cv2.COLOR_BGR2GRAY)

    num = cv2.getTrackbarPos('num','depth')
    blockSize = cv2.getTrackbarPos('blockSize','depth')
    if blockSize % 2 == 0:
        blockSize += 1
    if blockSize <5:
        blockSize = 5

    stereo = cv2.StereoBM_create(numDisparities = 16*num,blockSize = 31)
    disparity = stereo.compute(imgL,imgR)

    disp = cv2.normalize(disparity, disparity, alpha=0, beta=255,norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    threeD = cv2.reprojectImageTo3D(disparity.astype(np.float32)/16., camera_configs.Q)

    cv2.imshow('left', img1_rectified)
    cv2.imshow('right', img2_rectified)
    cv2.imshow('depth', disp)

    key = cv2.waitKey(20)
    if key == ord('q'):
        break
    elif key == ord('s'):
        #cv2.imwrite("path_BM_left.jpg", imgL)
        #cv2.imwrite("path_BM_left.jpg", imgR)
        cv2.imwrite("path_BM_depth.jpg", disp)
cap.release()
cv2.destroyAllWindows()


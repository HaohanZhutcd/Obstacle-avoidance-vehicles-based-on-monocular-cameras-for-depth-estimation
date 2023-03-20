import cv2
import numpy as np

left_camera_matrix = np.array([[452.21264,0,330.98428],[0,454.27970,247.41501],[0,0,1]])
left_distortion = np.array([[0.06936,-0.12084,0.00076,-0.00187,0.00000]])

right_camera_matrix = np.array([[452.21242,0,330.98337],[0,454.27941,247.41501],[0,0,1]])
right_distortion = np.array([[0.06936,-0.12084,0.00076,-0.00187,0.00000]])

om = np.array([0.0,0.0,0.0])
R = cv2.Rodrigues(om)[0]
T = np.array([-60.00000,0.00007,-0.00020])
size = (640,480)  #open windows size
R1,R2,P1,P2,Q,validPixROI1, validPixROI2 = cv2.stereoRectify(left_camera_matrix, left_distortion, right_camera_matrix,right_distortion,size,R,T)
left_map1,left_map2 = cv2.initUndistortRectifyMap(left_camera_matrix,left_distortion,R1,P1,size,cv2.CV_16SC2)
right_map1, right_map2 = cv2.initUndistortRectifyMap(right_camera_matrix,right_distortion,R2,P2,size,cv2.CV_16SC2)

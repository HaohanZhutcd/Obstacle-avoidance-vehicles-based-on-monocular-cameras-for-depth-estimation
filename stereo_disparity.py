# -*- coding: utf-8 -*-
"""
Created on Wed Dec 30 21:04:06 2020

@author: gkm0120
"""
import numpy as np
import cv2

# Reading the left and right images.

imgL = cv2.imread("/home/pi/car2/data/L/1.jpg",0)
imgR = cv2.imread("/home/pi/car2/data/R/1.jpg",0)

# Setting parameters for StereoSGBM algorithm
'''
minDisparity = 0;
numDisparities = 64;
blockSize = 11;
disp12MaxDiff = 1;
uniquenessRatio = 10;
speckleWindowSize = 10;
speckleRange = 8;
'''
minDisparity = 16;
numDisparities = 160 - 16;
blockSize = 3;
disp12MaxDiff = 1;
uniquenessRatio = 10;
speckleWindowSize = 100;
speckleRange = 2;
P1 = 8 * blockSize * blockSize
P2 = 32 * blockSize * blockSize
# Creating an object of StereoSGBM algorithm
stereo = cv2.StereoSGBM_create(
    minDisparity = minDisparity,
    numDisparities = numDisparities,
    blockSize = blockSize,
    P1 = P1,
    P2 = P2,
    disp12MaxDiff = disp12MaxDiff,
    uniquenessRatio = uniquenessRatio,
    speckleWindowSize = speckleWindowSize,
    speckleRange = speckleRange
)

# Calculating disparith using the StereoSGBM algorithm
disp = stereo.compute(imgL, imgR).astype(np.float32)
disp = cv2.normalize(disp,0,255,cv2.NORM_MINMAX)

# Displaying the disparity map
cv2.imshow("disparity",disp)
cv2.waitKey(0)
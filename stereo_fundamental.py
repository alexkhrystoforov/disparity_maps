import cv2
import numpy as np
from .utils import *


imgL = cv2.imread('data/set1_1.jpeg')
imgR = cv2.imread('data/set1_3.jpeg')

imgLgray = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
imgRgray = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)

h, w = imgLgray.shape

sift = cv2.SIFT_create()
matcher = Matcher(sift, imgLgray, imgRgray)
best_matches = matcher.BF_matcher()

print(len(best_matches))

pts1 = np.float32([matcher.kp2[m.queryIdx].pt for m in best_matches]).reshape(-1, 1, 2)
pts2 = np.float32([matcher.kp1[m.trainIdx].pt for m in best_matches]).reshape(-1, 1, 2)

F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC, 2, 0.99, 2000)

pts1 = pts1[mask.ravel() == 1]
pts2 = pts2[mask.ravel() == 1]

# get homography
ret, H1, H2 = cv2.stereoRectifyUncalibrated(pts1, pts2, F, (h, w))

# get rectified imgs

imgLrec = cv2.warpPerspective(imgL,H1,(w+200,h+200))
imgRrec = cv2.warpPerspective(imgR,H2,(w+200,h+200))

# params for StereoSGBM

max_disparity = 5*16
min_disparity = 0
num_disparities = 5*16
window_size = 5
stereo = cv2.StereoSGBM_create(minDisparity=min_disparity, numDisparities=num_disparities, blockSize=window_size,
                               uniquenessRatio=10, speckleWindowSize=100, speckleRange=2, disp12MaxDiff=12,
                               P1=8 * 3 * window_size, P2=32 * 3 * window_size)

# Post filtering

stereo2 = cv2.ximgproc.createRightMatcher(stereo)

lamb = 8000
sig = 1.5
visual_multiplier = 1.0
wls_filter = cv2.ximgproc.createDisparityWLSFilter(stereo)
wls_filter.setLambda(lamb)
wls_filter.setSigmaColor(sig)

disparity = stereo.compute(imgLrec, imgRrec)

disparity2 = stereo2.compute(imgRrec, imgLrec)
disparity2 = np.int16(disparity2)

filteredImg = wls_filter.filter(disparity, imgL, None, disparity2)
_, filteredImg = cv2.threshold(filteredImg, 0, max_disparity, cv2.THRESH_TOZERO)

filteredImg = (filteredImg / 16).astype(np.uint8)
stack = np.concatenate((imgLrec, imgRrec), axis=1)

cv2.namedWindow('imgL', cv2.WINDOW_NORMAL)
cv2.namedWindow('imgR', cv2.WINDOW_NORMAL)
cv2.namedWindow('disparity', cv2.WINDOW_NORMAL)
cv2.namedWindow('filtered', cv2.WINDOW_NORMAL)

cv2.imshow('disparity', disparity)
cv2.imshow('filtered', filteredImg)
cv2.imshow('imgL', imgLrec)
cv2.imshow('imgR', imgRrec)

cv2.waitKey(0)
cv2.imwrite('result_fundamental.png', filteredImg)

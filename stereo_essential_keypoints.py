import cv2
import numpy as np
from .utils import *


with np.load('iphone_photo.npz') as X:
    mtx, dist = [X[i] for i in ('name1', 'name2')]

imgL = cv2.imread('data/set1_1.jpeg')
imgR = cv2.imread('data/set1_3.jpeg')

imgLgray = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
imgRgray = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)

h, w = imgLgray.shape

sift = cv2.SIFT_create()
matcher = Matcher(sift, imgLgray, imgRgray)
best_matches = matcher.BF_matcher()

pts1 = np.float32([matcher.kp2[m.queryIdx].pt for m in best_matches]).reshape(-1, 1, 2)
pts2 = np.float32([matcher.kp1[m.trainIdx].pt for m in best_matches]).reshape(-1, 1, 2)

# print(pts1)

E, mask = cv2.findEssentialMat(pts1, pts2, mtx, method=cv2.RANSAC, prob=0.999, threshold=3.0, mask=None)

print('essential - \n', E)

pts1 = pts1[mask.ravel() == 1]
pts2 = pts2[mask.ravel() == 1]

# R - matrix, t - vector

_, R, t, mask = cv2.recoverPose(E, pts1, pts2, mtx)
print('rotation matrix - \n', R)
print('translation vector -\n ', t)
new_t = np.linalg.norm(t)
print(new_t)

# R1,R2 - rotation matricies, P1,P2 - projection matricies, Q - disparity to depth mapping matrix

R1, R2, P1, P2, Q, a, b = cv2.stereoRectify(mtx, dist, mtx, dist, (w, h + 100), R, t)

map1, map2 = cv2.initUndistortRectifyMap(mtx, dist, R1, P1, (w, h + 100), cv2.CV_16SC2)
imgLrec = cv2.remap(imgL, map1, map2, cv2.INTER_CUBIC)

map3, map4 = cv2.initUndistortRectifyMap(mtx, dist, R2, P2, (w, h + 100), cv2.CV_16SC2)

imgRrec = cv2.remap(imgR, map3, map4, cv2.INTER_CUBIC)

max_disparity = 5*16
min_disparity = -1
num_disparities = 5*16
window_size = 5
stereo = cv2.StereoSGBM_create(minDisparity=min_disparity, numDisparities=num_disparities, blockSize=window_size,
                               uniquenessRatio=10, speckleWindowSize=50, speckleRange=2, disp12MaxDiff=12,
                               P1=8 * 3 * window_size, P2=32 * 3 * window_size)


# POST FILTERING

stereo2 = cv2.ximgproc.createRightMatcher(stereo)

lamb = 8000
sig = 1.5
visual_multiplier = 1.0
wls_filter = cv2.ximgproc.createDisparityWLSFilter(stereo)
wls_filter.setLambda(lamb)
wls_filter.setSigmaColor(sig)

disparity = stereo.compute(imgLrec, imgRrec)

# Normalize the image
# min = disparity.min()
# max = disparity.max()
# disparity = np.uint8(255 * (disparity - min) / (max - min))

disparity2 = stereo2.compute(imgRrec, imgLrec)
# disparity2 = np.int16(disparity2)
disparity2 = np.float32(disparity2)

filteredImg = wls_filter.filter(disparity, imgL, None, disparity2)
_, filteredImg = cv2.threshold(filteredImg, 0, max_disparity, cv2.THRESH_TOZERO)
filteredImg = (filteredImg / 16).astype(np.uint8)

stack = np.concatenate((imgLrec,imgRrec), axis=1)

cv2.namedWindow('imgL', cv2.WINDOW_NORMAL)
cv2.namedWindow('imgR', cv2.WINDOW_NORMAL)
cv2.namedWindow('filtered', cv2.WINDOW_NORMAL)
cv2.namedWindow('d', cv2.WINDOW_NORMAL)
# cv2.namedWindow('stack', cv2.WINDOW_NORMAL)

# cv2.imshow('stack', stack)
cv2.imshow('imgL', imgLrec)
cv2.imshow('imgR', imgRrec)
cv2.imshow('filtered', filteredImg)
cv2.imshow('d', disparity)
cv2.waitKey(0)

cv2.imwrite('result_essential_keypoints.png', filteredImg)

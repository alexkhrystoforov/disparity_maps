import cv2
import numpy as np


with np.load('iphone_photo.npz') as X:
    mtx, dist = [X[i] for i in ('name1', 'name2')]

imgL = cv2.imread('data/left.jpeg')
imgR = cv2.imread('data/center.jpeg')

imgLgray = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
imgRgray = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

objp = np.zeros((14 * 19, 3), np.float32)
objp[:, :2] = np.mgrid[0:19, 0:14].T.reshape(-1, 2)

ret, cornersL = cv2.findChessboardCorners(imgLgray, (19,14))
pts1 = None
pts2 = None

if ret:
    print('success')
    corners2 = cv2.cornerSubPix(imgLgray, cornersL, (19, 14), (-1, -1), criteria)
    pts1 = np.float32(corners2)
    print(pts1)

ret, cornersR = cv2.findChessboardCorners(imgRgray, (19,14))

if ret:
    print('success')
    corners2 = cv2.cornerSubPix(imgRgray, cornersR, (19, 14), (-1, -1), criteria)
    pts2 = np.float32(corners2)

h, w = imgLgray.shape

E, mask = cv2.findEssentialMat(pts1, pts2, mtx, method=cv2.RANSAC, prob=0.999, threshold=3.0, mask=None)

print('essential - \n', E)

pts1 = pts1[mask.ravel() == 1]
pts2 = pts2[mask.ravel() == 1]

# R - matrix, t - vector

_, R, t, mask = cv2.recoverPose(E, pts1, pts2, mtx)
print('rotation matrix - \n', R)
print('translation vector -\n ', t)
new_t = np.linalg.norm(t) * 1.1
print(new_t)

# R1,R2 - rotation matricies, P1,P2 - projection matricies, Q - disparity to depth mapping matrix

R1, R2, P1, P2, Q, a, b = cv2.stereoRectify(mtx, dist, mtx, dist, (w+200, h+100), R, t)

map1, map2 = cv2.initUndistortRectifyMap(mtx, dist, R1, P1, (w+200, h+100), cv2.CV_16SC2)
imgLrec = cv2.remap(imgL, map1, map2, cv2.INTER_CUBIC)

map3, map4 = cv2.initUndistortRectifyMap(mtx, dist, R2, P2, (w+200, h+100), cv2.CV_16SC2)

imgRrec = cv2.remap(imgR, map3, map4, cv2.INTER_CUBIC)

max_disparity = 5*16
min_disparity = -1
num_disparities = 5*16
window_size = 5
stereo = cv2.StereoSGBM_create(minDisparity=min_disparity, numDisparities=num_disparities, blockSize=window_size,
                               uniquenessRatio=10, speckleWindowSize=100, speckleRange=2, disp12MaxDiff=12,
                               P1=8 * 3 * window_size, P2=32 * 3 * window_size)

stereo2 = cv2.ximgproc.createRightMatcher(stereo)

lamb = 8000
sig = 1.5
visual_multiplier = 1.0
wls_filter = cv2.ximgproc.createDisparityWLSFilter(stereo)
wls_filter.setLambda(lamb)
wls_filter.setSigmaColor(sig)

disparity = stereo.compute(imgLrec, imgRrec).astype(np.uint8)

disparity2 = stereo2.compute(imgRrec, imgLrec)

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
cv2.imwrite('result_essential_chess.png', filteredImg)
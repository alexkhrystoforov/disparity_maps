import cv2
import matplotlib.pyplot as plt
import numpy as np


with np.load('iphone_photo.npz') as X:
    mtx, dist = [X[i] for i in ('name1', 'name2')]


imgL = cv2.imread('left.jpeg', 0)
imgR = cv2.imread('right.jpeg', 0)


cv2.namedWindow('imgL', cv2.WINDOW_NORMAL)
cv2.namedWindow('imgR', cv2.WINDOW_NORMAL)

cv2.imshow('imgL', imgL)
cv2.imshow('imgR', imgR)

cv2.waitKey(0)

h, w = imgL.shape[:2]

imgL_und = cv2.undistort(imgL, mtx, dist, None)
imgR_und = cv2.undistort(imgR, mtx, dist, None)

cv2.namedWindow('imgL_und', cv2.WINDOW_NORMAL)
cv2.namedWindow('imgR_und', cv2.WINDOW_NORMAL)

cv2.imshow('imgL_und', imgL_und)
cv2.imshow('imgR_und', imgR_und)

cv2.waitKey(0)

stereo = cv2.StereoBM_create(numDisparities=64, blockSize=15)

disparity_compute = stereo.compute(imgL, imgR)

thr = cv2.threshold(disparity_compute, 0.6, 1.0, cv2.THRESH_BINARY)[1]
kernel = np.ones((3,3), np.uint8)
morph = cv2.morphologyEx(thr, cv2.MORPH_OPEN, kernel)

cv2.namedWindow('disp', cv2.WINDOW_NORMAL)
cv2.imshow('disp', disparity_compute)
cv2.namedWindow('thr', cv2.WINDOW_NORMAL)
cv2.imshow('thr', thr)
cv2.namedWindow('morph', cv2.WINDOW_NORMAL)
cv2.imshow('morph', morph)

cv2.waitKey(0)
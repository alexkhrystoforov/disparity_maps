import cv2
import numpy as np


imgL = cv2.imread('data/set1_1.jpeg')
imgR = cv2.imread('data/set1_3.jpeg')

imgLgray = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
imgRgray = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)

h, w = imgLgray.shape

stereo = cv2.StereoBM_create(numDisparities=3*16, blockSize=11)
disparity = stereo.compute(imgLgray, imgRgray)

# Normalize the image for representation
# min = disparity.min()
# max = disparity.max()
# disparity = np.uint8(255 * (disparity - min) / (max - min))

cv2.namedWindow('disparity', cv2.WINDOW_NORMAL)
cv2.imshow('disparity', disparity)

cv2.waitKey(0)

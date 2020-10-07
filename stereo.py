import numpy as np
import cv2
import glob
# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((14*19,3), np.float32)
objp[:,:2] = np.mgrid[0:19,0:14].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpointsL = [] # 3d point in real world space
imgpointsL = [] # 2d points in image plane.
objpointsR = []
imgpointsR = []

images = glob.glob('left.jpeg')

for fname in images:
    imgL = cv2.imread(fname)
    grayL = cv2.cvtColor(imgL,cv2.COLOR_BGR2GRAY)
    # Find the chess board corners
    ret, cornersL = cv2.findChessboardCorners(grayL, (19, 14),None)
    # If found, add object points, image points (after refining them)
    if ret == True:
        objpointsL.append(objp)
        cv2.cornerSubPix(grayL,cornersL,(11,11),(-1,-1),criteria)
        imgpointsL.append(cornersL)
        #cv2.namedWindow('imgl', cv2.WINDOW_NORMAL)
        #cv2.imshow('imgl',imgL)
        #cv2.waitKey(0)

ret, M1, d1, rvecs1, tvecs1 = cv2.calibrateCamera(objpointsL, imgpointsL,grayL.shape[::-1],None, None )

images = glob.glob('right.jpeg')

for fname in images:
    imgR = cv2.imread(fname)
    grayR = cv2.cvtColor(imgR,cv2.COLOR_BGR2GRAY)
    # Find the chess board corners
    ret, cornersR = cv2.findChessboardCorners(grayR, (19,14),None)
    # If found, add object points, image points (after refining them)
    if ret == True:
        objpointsR.append(objp)
        cv2.cornerSubPix(grayR,cornersR,(11,11),(-1,-1),criteria)
        imgpointsR.append(cornersR)
        #cv2.namedWindow('imgR', cv2.WINDOW_NORMAL)
        #cv2.imshow('imgR', imgR)
        #cv2.waitKey(0)


ret, M2, d2, rvecs2, tvecs2 = cv2.calibrateCamera(objpointsR, imgpointsR,grayR.shape[::-1],None, None )
# R - rotation matrix, T - translation vector, E - essential matrix, F - fundamental matrix
retval, M1, d1, M2, d2, R, T, E, F = cv2.stereoCalibrate(objpointsL, imgpointsL, imgpointsR, M1, d1, M2, d2, (1276,956))
print(E)

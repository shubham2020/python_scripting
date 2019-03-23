#!/usr/bin/env python

import numpy as np
import cv2 as cv
import glob
# Load previously saved data
with np.load('calibrationMatrices.npz') as X:
    mtx, dist, _, _ = [X[i] for i in ('mtx','dist','rvecs','tvecs')]

#Now let's create a function, draw which takes the corners in the chessboard (obtained using cv.findChessboardCorners()) and axis points to draw a 3D axis.

def draw(img, corners, imgpts):
    corner = tuple(corners[0].ravel())
    img = cv.line(img, corner, tuple(imgpts[0].ravel()), (255,0,0), 5)
    img = cv.line(img, corner, tuple(imgpts[1].ravel()), (0,255,0), 5)
    img = cv.line(img, corner, tuple(imgpts[2].ravel()), (0,0,255), 5)
    return img

criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
objp = np.zeros((7*7,3), np.float32)
objp[:,:2] = np.mgrid[0:13.2:7j,0:13.2:7j].T.reshape(-1,2)
axis = np.float32([[6.6,0,0], [0,6.6,0], [0,0,-6.6]]).reshape(-1,3)

#Now, as usual, we load each image.
#Search for 7x7 grid. If found, we refine it with subcorner pixels.
#Then to calculate the rotation and translation, we use the function, cv.solvePnPRansac().
#Once we those transformation matrices, we use them to project our axis points to the image plane.
# In simple words, we find the points on image plane corresponding to each of (6.6,0,0),(0,6.6,0),(0,0,-6.6) in 3D space. Once we get them, we draw lines from the first corner to each of these points using our draw() function.
#Done!!!

for fname in glob.glob('/home/shubham/Pictures/chessboard/*.jpg'):
    img = cv.imread(fname)
    gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    ret, corners = cv.findChessboardCorners(gray, (7,7),None)
    if ret == True:
        corners2 = cv.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        # Find the rotation and translation vectors.
        ret,rvecs, tvecs = cv.solvePnP(objp, corners2, mtx, dist)
        # project 3D points to image plane
        imgpts, jac = cv.projectPoints(axis, rvecs, tvecs, mtx, dist)
        img = draw(img,corners2,imgpts)
        cv.imshow('img',img)
        k = cv.waitKey(0) & 0xFF
        if k == ord('s'):
            cv.imwrite(fname[:6]+'.png', img)
cv.destroyAllWindows()

#!/usr/bin/env python

import numpy as np
import cv2 as cv
import glob
# termination criteria for cornerSubPix
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((7*7,3), np.float32)
#objp[:,:2] = np.mgrid[0:7,0:7].T.reshape(-1,2) # -1 so that the other dimension is automatically decided
objp[:,:2] = np.mgrid[0:13.2:7j,0:13.2:7j].T.reshape(-1,2)
#use this if dimension of the each box of chessboard is known :- np.mgrid[0:240:9j,0:240:9j].T.reshape(-1,2)
# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

images = glob.glob('/home/shubham/Pictures/chessboard/*.jpg')
for fname in images:
    img = cv.imread(fname)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # Find the chess board corners
    ret, corners = cv.findChessboardCorners(gray, (7,7), None)
    #print(corners[1])
    # If found, add object points, image points (after refining them because findChessboardCorners function returs approximate coordinates of the corner)
    if ret == True:
        objpoints.append(objp)
        corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners)
        #print(corners2[1])
        # Draw and display the corners
        #cv.drawChessboardCorners(img, (7,7), corners2, ret)
        #cv.imshow('img', img)
        #cv.waitKey(500)
cv.destroyAllWindows()

#To get the intrinsic and extrinsic parameter matrices
ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
#np.savez('calibrationMatrices', mtx=mtx, dist=dist, rvecs=rvecs, tvecs=tvecs) #the arrays are saved with the keyword names
#print("Intrinsic Parameter Matrix :- \n", mtx)
#print("Distortion Matrix :- \n", dist )
#print("Rotation Matrix :- \n", rvecs[0])
#print("Translation Matrix :- \n", rvecs[0])
#print(gray.shape)

# we can refine the camera matrix based on a free scaling parameter using cv.getOptimalNewCameraMatrix().
#If the scaling parameter alpha=0, it returns undistorted image with minimum unwanted pixels. #
#So it may even remove some pixels at image corners.
#If alpha=1, all pixels are retained with some extra black images.
img = cv.imread(images[1])
h,  w = img.shape[:2]
newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h)) #here 1 is for alpha and right (w,h) is for image size after rectification
#print("New Camera Matrix after optimizing :- ",newcameramtx)
#print("Region of interest :- ", roi)

#Now, we can take an image and undistort it. OpenCV comes with two methods for doing this.

#Using cv.undistort()

# undistort
dst = cv.undistort(img, mtx, dist, None, newcameramtx)
# crop the image
x, y, w, h = roi
dst = dst[y:y+h, x:x+w]
#cv.imwrite('calibresult.png', dst)
cv.imshow('img', dst)
cv.waitKey(10000)

#Using remapping

# undistort
#mapx, mapy = cv.initUndistortRectifyMap(mtx, dist, None, newcameramtx, (w,h), 5)
#dst = cv.remap(img, mapx, mapy, cv.INTER_LINEAR)
# crop the image
#x, y, w, h = roi
#dst = dst[y:y+h, x:x+w]
#cv.imwrite('calibresult.png', dst)
#cv.imshow('img', dst)
#cv.waitKey(10000)

#Reprojection error
#Re-projection error gives a good estimation of just how exact the found parameters are.
#Given the intrinsic, distortion, rotation and translation matrices, we must first transform the object point to image point using cv.projectPoints().
#Then, we can calculate the absolute norm between what we got with our transformation and the corner finding algorithm.
#To find the average error, we calculate the arithmetical mean of the errors calculated for all the calibration images.

mean_error = 0
for i in xrange(len(objpoints)):
    imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2)/len(imgpoints2)
    mean_error += error
print( "total error: {}".format(mean_error/len(objpoints)) )

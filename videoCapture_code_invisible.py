#!/usr/bin/env python
#VideoCapture from mobile camera usign droidcam module and ADB support
#################################################################

#Author : Shubham Pathak

#################################################################

import numpy as np
import cv2 as cv


url = "http://lenovop2:@exergy@172.16.181.91:4747/mjpegfeed?960x720"#"protocol://username:password@IP:port/feedformat?resolution"
cap = cv.VideoCapture(url)
if not cap.isOpened():
    print("Cannot open camera")
    exit()
ret, frame_old = cap.read()
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    # Our operations on the frame come here
    #gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    #edges = cv.Canny(frame,100,200)
    # Convert BGR to HSV
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    # define range of blue color in HSV
    lower_blue = np.array([90,50,50])
    upper_blue = np.array([130,255,255])
    # Threshold the HSV image to get only blue colors
    mask = cv.inRange(hsv, lower_blue, upper_blue)
    # Bitwise-AND mask and original image
    res = cv.bitwise_and(frame,frame, mask= mask)
    substract = frame - res #removes blue color from the fram and leaves all others
    old = cv.bitwise_and(frame_old,frame_old,mask = mask)
    new = cv.add(substract,old)
    # Display the resulting frame
    cv.imshow('frame', new)
    if cv.waitKey(1) == ord('q'):
        break
# When everything done, release the capture
cap.release()
cv.destroyAllWindows()

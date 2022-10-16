import cv2 as cv
import numpy as np 
import imutils
from imutils.video import VideoStream
import time

blueLower = (100, 150, 100)
blueUpper = (110, 255, 255)
vs = VideoStream(src=0).start()
time.sleep(2.0)
while True:
    frame = vs.read()
    frame = imutils.resize(frame, width=600)
    #same way it is done for the ball
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    mask = cv.inRange(hsv, blueLower, blueUpper)
    mask = cv.erode(mask, None, iterations=2)
    mask = cv.dilate(mask, None, iterations=2)
    cv.imshow('mask',mask)
    key = cv.waitKey(1) & 0xFF
    if key == ord("q"):
        break

cv.destroyAllWindows

# REPLACE CONTOUR MAPPING WITH KMEANS CLUSTERING
# WARNING THIS IS NOT THE FINAL DESIGN. THIS USES CONTOUR MAPPING
from pydoc import doc
import cv2 as cv
import numpy as np 
import imutils
from imutils.video import VideoStream
#this is for the intel stereo depth camera
import pyrealsense2 as rs
import time

#this is for the adafruit servo and motor controler board
from board import SCL, SDA
import busio
from adafruit_extended_bus import ExtendedI2C as I2C
from adafruit_motor import servo
from adafruit_pca9685 import PCA9685

def real_time_shape(show):
    # VIDEO CAPTURE
    #cap_video = cv.VideoCapture(0)
    #setting color limits for the mask
    blueLower = (100, 150, 100)
    blueUpper = (110, 255, 255)
    vs = VideoStream(src=0).start()
    time.sleep(2.0)
    #setting up the stereo deapth camera
    pipeline = rs.pipeline()
    config = rs.config()

    # Get device product line for setting a supporting resolution for stereo depth
    pipeline_wrapper = rs.pipeline_wrapper(pipeline)
    pipeline_profile = config.resolve(pipeline_wrapper)
    device = pipeline_profile.get_device()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    pipeline.start(config)
    #setting up the servo and motor controller
    #i2c = I2C(8)
    i2c = busio.I2C(SCL, SDA)
    # Create a simple PCA9685 class instance.
    pca = PCA9685(i2c)
    pca.frequency = 50
    servo1 = servo.Servo(pca.channels[0], actuation_range=10) #this will be how I deal with turning, 0 is right, 5 is straight, 10 is left
    servo2 = servo.Servo(pca.channels[1],min_pulse = 400,max_pulse=2400)#same as before write 80 to intialize/set motors to offf, 100 seems about right for driving if it is jnot i will make the range smaller
    servo2.angle = 80
    time.sleep(4)
    # RUNS FOREVER
    while(1):
        #reading the normal camera
        frame = vs.read()
        framers = pipeline.wait_for_frames()
        depth_frame = framers.get_depth_frame()
        #same way it is done for the ball, applies a mask and makes it smaller
        hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        mask = cv.inRange(hsv, blueLower, blueUpper)
        mask = cv.erode(mask, None, iterations=2)
        mask = cv.dilate(mask, None, iterations=2)
        #this is to test for the size of the frame
        # height,width =frame.shape[:2]
        # print("height=",height,"Width=",width),
        # time.sleep(20)


        
        

        # CANNY EDGE DETECTION, I am not currently using this 

        frameG = cv.cvtColor(frame,cv.COLOR_RGB2GRAY)
        edges = cv.Canny(frameG,200,200)
        thresh = cv.threshold(frameG,127,255,cv.THRESH_BINARY)[1]



        # CALLING SHAPE DETECTION FUNCTION
        try:
            shapes,x,y,shape = shapeDetector(mask,frame.copy())
            #below pulls the center of the shape and takes the average deapth of the 10x10 pixle area around it  
            XL = createList(x-5,x+5)
            YL = createList(y-5,y+5)
            zDepthtot = 0
            avedepth = 0
            divisor = 0
            #zDepth = depth_frame.get_distance(int(x),int(y))
            for xx in XL:
                for yy in YL:
                    zDepthtot =zDepthtot+ depth_frame.get_distance(int(xx),int(yy))
                    if depth_frame.get_distance(int(xx),int(yy)) != 0:
                        divisor = divisor +1
            avedepth = zDepthtot/divisor
        except:
            pass

        # if (show):
        #     # DISPLAY ORIGINAL
        #     cv.imshow('Original Image',frame)

            
        #     # DISPLAY SHAPE OUTPUT
        #     try:
        #         cv.imshow('Shapes',shapes)
        #     except:
        #         pass
                #print("no shape in view")
        #this is where the movement code and instructions will go, starting assuming that the robot is in front of it and can see it
        if avedepth <.5:
            servo2.angle = 80
            servo1.angle = 5
            break
        else:# avedepth <2:
            if avedepth <1: #this is my final approach code 
                if 240<=x<=440 and shape =='square':
                    servo1.angle = 5
                    servo2.angle = 100
                else:
                    servo1.angle = 5
                    servo2.angle = 70 #hopefully this makes it back up
            else:
                if avedepth >1:
                    #print("its trying to write")
                    if 240<=x<=440:
                        servo1.angle = 5
                        servo2.angle = 100
                    elif 0<=x<=240:
                        servo1.angle = 10
                        servo2.angle = 100
                    else:# 440< x <640:
                        servo1.angle = 0
                        servo2.angle = 100



        # else:
        #     do
        
        cv.waitKey(5)
    #killing pipelines and videos
    vs.stop()
    pipeline.stop()
    # cv.destroyAllWindows()
#simple function to crate a list from number r1 to number r2
def createList(r1, r2):
    return list(range(r1, r2+1))

def shapeDetector(image,origimage):
    
    # RESIZING THE IMAGE
    resized = imutils.resize(image, width=300)
    ratio = image.shape[0] / float(resized.shape[0])
    
    # SETTING A THRESHOLD TO CONVERT IT TO BLACK AND WHITE

    # FINDING CONTOURS IN THE B/W IMAGE
    contours = cv.findContours(resized.copy(),cv.RETR_EXTERNAL,cv.CHAIN_APPROX_NONE)[0]
    
    if len(contours)>0:
        #this only takes the largest contour, we do not want to deal with other small contours messing us up
        cn = max(contours, key = cv.contourArea)

        # for cntour in contours:
            # CALCULATING THE CENTERgray = cv.cvtColor(resized, cv.COLOR_BGR2GRAY)
        
        shape = detect(cn)#tour)
        #finds the center, resizes, and draws the contours on the image, could lose some of this as it is not for viewing
        M = cv.moments(cn)#tour)
        if (M["m00"] == 0):
            cX = 0
            cY = 0
        else:
            cX = int((M["m10"] / M["m00"]) * ratio)
            cY = int((M["m01"] / M["m00"]) * ratio)
        cntour = cn.astype("float")
        cn = cn*ratio
        cn = cn.astype("int")
        cv.drawContours(origimage,[cn],-1,(34,0,156),2)
        cv.putText(origimage, shape, (cX, cY), cv.FONT_HERSHEY_SIMPLEX,
        0.5, (255, 255, 255), 2)
        return(origimage,cX,cY,shape)
    else:
        pass

    # NEEDS TO BE REPLACED BY K MEANS CLUSTERING INSTEAD OF CONTOUR MAPPING the last person wrote this, i am not sure how it would be applied


#this determines the shape based on the points
def detect(c):
    shape = "unidentified"
    peri = cv.arcLength(c,True)
    approx = cv.approxPolyDP(c, 0.04 * peri, True)
    if len(approx) == 3:
        shape = "triangle"

    elif len(approx) == 4:
        (_, _, w, h) = cv.boundingRect(approx)
        ar = w / float(h)
        shape = "square" if ar >= 0.95 and ar <= 1.05 else "rectangle"

    elif len(approx) == 5:
        shape = "pentagon"

    else:
        shape = "circle"

    return shape


if __name__ == "__main__":
    
    real_time_shape(1)
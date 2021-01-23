#!/usr/bin/env python

# Python 2/3 compatibility
#from __future__ import print_function

# Import necessary packages
import cv2 as cv
import numpy as np
import time
import os
import InitBackground
import DetectAndDraw
import VideoStream

## Camera settings
IM_WIDTH = 640 #1280
IM_HEIGHT = 480 #720 
FRAME_RATE = 10

## Initialize calculated frame rate because it's calculated AFTER the first time it's displayed
frame_rate_calc = 1
freq = cv.getTickFrequency()

## Define font to use
font = cv.FONT_HERSHEY_SIMPLEX

# Initialize camera object and video feed from the camera. The video stream is set up
# as a seperate thread that constantly grabs frames from the camera feed. 
# See VideoStream.py for VideoStream class definition
## IF USING USB CAMERA INSTEAD OF PICAMERA,
## CHANGE THE THIRD ARGUMENT FROM 1 TO 2 IN THE FOLLOWING LINE:
videostream = VideoStream.VideoStream((IM_WIDTH,IM_HEIGHT),FRAME_RATE,1,0).start()
time.sleep(1) # Give the camera time to warm up

if __name__ == '__main__':
    import sys, getopt
      
    'F'
    print(__doc__)

    args, video_src = getopt.getopt(sys.argv[1:], '', ['cascade=', 'nested-cascade='])
    try:
        video_src = video_src[0]
    except:
        video_src = 0
    args = dict(args)
    cascade_fn = args.get('--cascade', "data/haarcascades/haarcascade_frontalface_alt.xml")
    nested_fn  = args.get('--nested-cascade', "data/haarcascades/haarcascade_eye.xml")

    cascade = cv.CascadeClassifier(cv.samples.findFile(cascade_fn))
    nested = cv.CascadeClassifier(cv.samples.findFile(nested_fn))
    
    ' SUB '

    fgbg = cv.createBackgroundSubtractorMOG2()
    #Persobkg = InitBackground.InitBckGrnd(videostream)
    cam_quit = 0 # Loop control variable
    compteur = 0
    NbDetection = 0
    i = 0 #titr face
    f = 0 #nb photo / face
    t = 0 #Tmps limit

    while cam_quit == 0:
        
        img = videostream.read()
        # Start timer (for calculating frame rate)
        t1 = cv.getTickCount()
        
        'S'
        mask = InitBackground.MaskMOG2(img,fgbg)
        #mask = InitBackground.MaskPerso(img)
        #vis = None
        if NbDetection<1:
            nb = InitBackground.Nb(mask) #Fontionn mis trop gourmn
            #print("%bln = " + str(nb))
            if nb>5:
                print("BIPBIPBIP")
                NbDetection = NbDetection+1
            else:
                print("---")
                NbDetection = 0
                
            # Calculate framerate
            t2 = cv.getTickCount()
            time1 = (t2-t1)/freq
            frame_rate_calc = 1/time1
            
        else:
            #cv.imshow('cam',img)
            #cv.imshow('mask',mask)
            t = t + 1
            gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            gray = cv.equalizeHist(gray)
	    
            rects = DetectAndDraw.detect(gray, cascade)
            vis = img.copy()
            vis_roi = None
            DetectAndDraw.draw_rects(vis, rects, (0, 255, 0))
            if not nested.empty():
                for x1, y1, x2, y2 in rects:
                    roi = gray[y1:y2, x1:x2]
                    vis_roi = vis[y1:y2, x1:x2]
                    subrects = DetectAndDraw.detect(roi.copy(), nested)
                    DetectAndDraw.draw_rects(vis_roi, subrects, (255, 0, 0))
                    f = f + 1
            
            # Calculate framerate
            t2 = cv.getTickCount()
            time1 = (t2-t1)/freq
            frame_rate_calc = 1/time1
            
            # Draw framerate in the corner of the image. Framerate is calculated at the end of the main loop,
            # so the first time this runs, framerate will be shown as 0.
            cv.putText(vis,"FPS: "+str(int(frame_rate_calc)),(10,26),font,0.7,(255,0,255),2,cv.LINE_AA)
            
            
            #cv.imshow('facedetect', vis)
            if vis_roi is not None:
                #cv.imshow('last detected face', vis_roi)
                cv.imwrite("face"+str(i)+"f"+str(f)+".png", vis)
                if f == 3:
                    f = 0
                    i = i + 1
                    t = 0
                    NbDetection = 0
                    #cv.destroyAllWindows()
                    
            if t > 50:
                f = 0
                i = i + 1
                t = 0
                NbDetection = 0
        
        
        # Poll the keyboard. If 'q' is pressed, exit the main loop.
        key = cv.waitKey(1) & 0xFF
        if key == ord("q"):
            cam_quit = 1
            
    # Close all windows and close the PiCamera video stream.
    cv.destroyAllWindows()
    videostream.stop()


# Import necessary packages
import numpy as np
import cv2 as cv
import time


"Detection and draw face/eyes on current frame"
def detect(img, cascade):
    rects = cascade.detectMultiScale(img, scaleFactor=1.3, minNeighbors=4, minSize=(30, 30),
                                        flags=cv.CASCADE_SCALE_IMAGE)
    if len(rects) == 0:
        return []
    rects[:,2:] += rects[:,:2]
    return rects
    
def draw_rects(img, rects, color):
     for x1, y1, x2, y2 in rects:
          cv.rectangle(img, (x1, y1), (x2, y2), color, 2)
            

# Import necessary packages
import numpy as np
import cv2
import time
import VideoStream


"Detection and draw face/eyes on current frame"
def InitBckGrnd(videostream):
    
   cam_quit = 0 
   while cam_quit == 0:
       out = videostream.read()
       cam_quit = 1
       out = MaskPerso(out)
       
       return out

def MaskPerso(img):
    
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    
    # noise removal
    kernel = np.ones((3,3),np.uint8)
    opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)
    
    return opening
    
def MaskMOG2(img, bckgrndsub):
    
    fgmask = bckgrndsub.apply(img)
    
    return fgmask

def Detect(img, mask):
    
    vis = img.copy()
    #w,h = vis.size
    var = 0
    var = vis.any() - mask.any()
    if var != 0:
        var = 50
        
    for pix in vis:
        for pix2 in mask:
            vis[pix] = vis[pix] - mask[pix2]
            
                
                
    
    return vis

def Nb(vis):
    
    #vis = mask.copy()
    nb = 0
    for lin in vis:
            for pxl in lin:
                if pxl == 255:
                    nb = nb+1
    nb = 100*nb/ (640*480)
    #w,h = vis.size
    '''
    var = 0
    var = vis.any() - mask.any()
    if var != 0:
        var = 50
        
    for pix in vis:
        for pix2 in mask:
            vis[pix] = vis[pix] - mask[pix2]
            '''
                
                
    
    return nb



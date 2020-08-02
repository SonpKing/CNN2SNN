#!/usr/bin/env python
'''
Usage:
    ./ssearch.py input_image (f|q)
    f=fast, q=quality
Use "l" to display less rects, 'm' to display more rects, "q" to quit.
'''

import sys
import numpy as np
import cv2 as cv

class Camera:
    def __init__(self):
        self.cap = cv.VideoCapture(0)
        print("successfully open the camera")

    def ready(self):
        return self.cap.isOpened()

    def get_frame(self):
        # Capture frame-by-frame
        if not self.ready():
            print("cannot open the camera")
            return None 
        ret, frame = self.cap.read()
        # if frame is read correctly ret is True
        if not ret:
            return None
        else:
            return frame

    def display_video(self):
        if not self.ready():
            print("cannot open the camera")
            return 
        while True:
            frame = self.get_frame()
            if frame is not None:
                cv.imshow('frame',frame)
                if cv.waitKey(1) == ord('q'):
                    break
            else:
                print("Can't receive frame (stream end?). Exiting ...")
                break
        cv.destroyAllWindows()
    
    def __del__(self):
        self.cap.release()


def GaussianBlur(img):
    return cv.GaussianBlur(img, (5,5),0)

def Resize(img, newHeight=500):
    newWidth = int( img.shape[1] * newHeight / img.shape[0] )
    img = cv.resize( img, (newWidth, newHeight) )
    img = img[int(newHeight * 0.1): int(newHeight * 0.9), int(newWidth * 0.1): int(newWidth * 0.9)]
    return img


class SelectiveSearch:
    def __init__(self, threads=1):
        cv.setUseOptimized( True )
        cv.setNumThreads(threads)
        self.process = []
        self.set_preprocess(GaussianBlur)
        self.quality = True

    def set_preprocess(self, func):
        #func shouldn't 
        self.process.append(func)

    def preprocess(self, img):
        img = img.copy()
        for process in self.process:
            img = process(img)
        return img

    def select(self, img):
        img = self.preprocess(img)
        # create Selective Search Segmentation Object using default parameters
        ss = cv.ximgproc.segmentation.createSelectiveSearchSegmentation()
        # set input image on which we will run segmentation
        ss.setBaseImage(img)
        cv.imshow( "Output", img )
        if self.quality:
            ss.switchToSelectiveSearchQuality()
        else:
            ss.switchToSelectiveSearchFast()
        # run selective search segmentation on input image
        rects = ss.process()
        print( 'Total Number of Region Proposals: {}'.format( len( rects ) ) )
        return rects

def generate_bb(img, rect, resize=32):
    x, y, w, h = rect
    img = img[x : x + w, y : y + h]
    img = cv.resize(img, (resize, resize))
    return np.array(img) / 255.0

    

if __name__ == '__main__':
    cam = Camera()
    searcher = SelectiveSearch()
    img = cam.get_frame()
    img = Resize(img)
    rects = searcher.select(img)

    # number of region proposals to show
    numShowRects = 5
    # increment to increase/decrease total number
    # of reason proposals to be shown
    increment = 5

    while True:
        # create a copy of original image
        imOut = img.copy()

        # itereate over all the region proposals
        for i, rect in enumerate( rects ):
            # draw rectangle for region proposal till numShowRects
            if (i < numShowRects):
                x, y, w, h = rect
                cv.rectangle( imOut, (x, y), (x + w, y + h), (0, 255, 0), 1, cv.LINE_AA )
            else:
                break

        # show output
        cv.imshow( "Output", imOut )

        # record key press
        k = cv.waitKey( 0 ) & 0xFF

        # m is pressed
        if k == 109:
            # increase total number of rectangles to show by increment
            numShowRects += increment
        # l is pressed
        elif k == 108 and numShowRects > increment:
            # decrease total number of rectangles to show by increment
            numShowRects -= increment
        # q is pressed
        elif k == 113:
            break
        print("total number to show:",  numShowRects)
    # close image show window
    cv.destroyAllWindows()

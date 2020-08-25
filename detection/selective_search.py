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
from util.util import Timer

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

def Resize(img, newHeight=400):
    newWidth = int( img.shape[1] * newHeight / img.shape[0] )
    img = cv.resize( img, (newWidth, newHeight) )
    # img = img[int(newHeight * 0.1): int(newHeight * 0.9), int(newWidth * 0.1): int(newWidth * 0.9)]
    return img


class SelectiveSearch:
    def __init__(self, threads=1, quality=1):
        cv.setUseOptimized( True )
        cv.setNumThreads(threads)
        self.process = []
        self.set_preprocess(GaussianBlur)
        self.quality = quality

    def set_preprocess(self, func):
        #func shouldn't 
        self.process.append(func)

    def preprocess(self, img):
        img = img.copy()
        for process in self.process:
            img = process(img)
        return img

    def select(self, img):
        timer = Timer()
        img = self.preprocess(img)
        # create Selective Search Segmentation Object using default parameters
        ss = cv.ximgproc.segmentation.createSelectiveSearchSegmentation()
        # set input image on which we will run segmentation
        ss.setBaseImage(img)
        # cv.imshow( "Output", img )
        if self.quality==1:
            ss.switchToSelectiveSearchQuality()
        elif self.quality==0:
            ss.switchToSelectiveSearchFast()
        else:
            ss.switchToSingleStrategy(100)
        # run selective search segmentation on input image
        rects = ss.process()
        timer.record('Total Number of Region Proposals: {}'.format( len( rects ) ) )
        return rects

def rect2box(rect):
    x, y, w, h = rect
    return (x, y, x + w, y + h)

def bb_isvalid(img, rect, ratio=4.0):
    _, _, w, h = rect
    return 1.0 / ratio < w / h < ratio and w > 20 and h > 20

def generate_bb(img, box, resize=32):
    x1, y1, x2, y2 = box
    assert x1 < x2 and y1 < y2
    img_bb = img[y1 : y2, x1 : x2]
    if resize:
        img_bb = cv.resize(img_bb, (resize, resize))
    return np.array(img_bb) / 255.0

def show_bb(img, boxes, numShowRects=10000, show_time=5000):
    imOut = img.copy()
    for i, box in enumerate( boxes ):
        # draw rectangle for region proposal till numShowRects
        if i == 0:
            x, y, x2, y2 = box
            cv.rectangle( imOut, (x, y), (x2, y2), (255, 0, 0), 1, cv.LINE_AA )
        elif (i < numShowRects):
            x, y, x2, y2 = box
            cv.rectangle( imOut, (x, y), (x2, y2), (0, 255, 0), 1, cv.LINE_AA )
        else:    
            break
    cv.imshow( "Output", imOut )
    cv.waitKey(show_time)
    

def vis_bb(img, bbox_pred, scores, cls_inds, class_name, show_time=0, show=False):
    img = img.copy()
    class_color = [(np.random.randint(255),np.random.randint(255),np.random.randint(255)) for _ in range(len(class_name))]
    for i, box in enumerate(bbox_pred):
        cls_indx = cls_inds[i]
        xmin, ymin, xmax, ymax = box
        box_w = int(xmax - xmin)
        # print(xmin, ymin, xmax, ymax)
        # newHeight = 1080
        # scale = newHeight / img.shape[0]
        # newWidth = int(img.shape[1] * scale)
        # xmin, ymin, xmax, ymax = xmin*scale, ymin*scale, xmax*scale, ymax*scale
        # cv.resize(img, (newWidth, newHeight))
        cv.rectangle(img, (int(xmin), int(ymin)), (int(xmax), int(ymax)), class_color[int(cls_indx)], 1)
        cv.rectangle(img, (int(xmin), int(ymin)), (int(xmin+box_w*0.55), int(ymin+15)), class_color[int(cls_indx)], -1)
        mess = '%s: %.3f' % (class_name[int(cls_indx)], scores[i])
        cv.putText(img, mess, (int(xmin), int(ymin+15)), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)
    if show==False:
        # cv.imshow("Result", img)
        # cv.waitKey(show_time)
        # cv.destroyAllWindows()
        return img
    else:
        b,g,r = cv.split(img) 
        img = cv.merge([r,g,b])
        from matplotlib import pyplot as plt 
        fig = plt.figure(figsize=(4, 2))
        ax1 = fig.add_subplot(111)
        ax1.imshow(img)
        ax1.axis('off')
        return fig
    


def filter_rects(img, rects, ratio=4.0, size=51):
    boxes = []
    for rect in rects:
        if bb_isvalid(img, rect, ratio):
            boxes.append(rect2box(rect))
            if len(boxes) >= size:
                break
        else:
            # print(rect)
            continue
    boxes = np.array(boxes)
    return boxes

if __name__ == '__main__':
    # cam = Camera()
    searcher = SelectiveSearch()
    # img = cam.get_frame()
    img = cv.imread("data/2.png")
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
                # print(rect)
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



    

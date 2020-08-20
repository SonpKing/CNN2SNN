import cv2 as cv
from .selective_search import SelectiveSearch, generate_bb, Resize, filter_rects
import os

def generate_img(file_path, save_path, step, seed=""):
    cap = cv.VideoCapture(file_path)
    ind = 0
    searcher = SelectiveSearch()
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    while cap.isOpened():
        ret, fram = cap.read()
        if ret and ind % step == 0:
            img = Resize(fram)
            rects = searcher.select(img)
            boxes = filter_rects(img, rects, ratio=3.0)
            for i, box in enumerate(boxes):
                img_bb = generate_bb(img, box, None)
                # cv.imshow("Output", img_bb)
                # cv.waitKey(1000)
                cv.imwrite(os.path.join(save_path, seed + str(ind)+"_"+str(i)+".png"), img_bb*255)
        ind += 1


# -*- coding: utf-8 -*-
from .WebDraw import WebDraw
import matplotlib.pyplot as plt
from random import randint
from time import sleep
import cv2 as cv

# def show_async(plt_que, ctl_que):
#     # initilize, PORT is websocket port
#     draw = WebDraw(PORT=8801)
#     print("web drawing started")

#     while True:
#         try:
#             if not ctl_que.empty():
#                 print("web drawing exit")
#                 break
#             while plt_que.empty():
#                 sleep(0.001)
#             fig = plt_que.get()
#             draw(fig)
#         except:
#             continue


def show_async(plt_que, ctl_que):
    window_name = 'Object Recognization'
    cv.namedWindow(window_name, cv.WND_PROP_FULLSCREEN)
    cv.setWindowProperty(window_name, cv.WND_PROP_FULLSCREEN, cv.WINDOW_FULLSCREEN)
    print("opencv showing started")
    img = cv.imread("data\\demo.png")
    while plt_que.empty():
        cv.imshow(window_name, img)
        cv.waitKey(1)

    while True:
        try:
            if not ctl_que.empty():
                print("web drawing exit")
                break
            while plt_que.empty():
                sleep(0.01)
            img = plt_que.get()
            cv.imshow(window_name, img)
            cv.waitKey(1)
        except:
            continue




    
    


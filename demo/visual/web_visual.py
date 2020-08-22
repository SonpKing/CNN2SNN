from .WebDraw import WebDraw
import matplotlib.pyplot as plt
from random import randint
from time import sleep
import cv2 as cv

def show_async(plt_que, ctl_que):
    # initilize, PORT is websocket port
    draw = WebDraw(PORT=8801)
    print("web drawing started")

    while True:
        try:
            if not ctl_que.empty():
                print("web drawing exit")
                break
            while plt_que.empty():
                sleep(0.01)
            fig = plt_que.get()
            draw(fig)
        except:
            continue


# def show_async(plt_que, ctl_que):
#     # initilize, PORT is websocket port
#     # draw = WebDraw(PORT=8801)
#     print("web drawing started")

#     while True:
#         try:
#             if not ctl_que.empty():
#                 print("web drawing exit")
#                 break
#             while plt_que.empty():
#                 sleep(0.01)
#             img = plt_que.get()
#             cv.imshow("Result", img)
#             cv.waitKey(0)
#         except:
#             continue




    
    


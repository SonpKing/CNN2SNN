from detection import Camera, Resize
from .comm import ClientConnection
import pickle, sys, struct
from time import sleep
import numpy as np
import cv2 as cv
from .detect_util import img2bytearray
import os


##TODO while first or connectiong firsts
def client_run(IP, Port):
    # cam = TmpCam("C:\\Users\\dell\\Desktop\\Detect_Data\\mytest")#Camera()
    cam = TmpCam("C:\\Users\\zjlab\\Desktop\\Detect_Data\\mytest")#Camera()
    application_id = 7
    buffer_num = 0
    with ClientConnection(IP, Port) as conn:
        print("detection client connected")
        while True:
            while buffer_num > 1:
                try:
                    _, data = conn.receive()
                    if data != None:
                        buffer_num -= 1
                except:
                    sleep(0.01)
            data = get_img_bytes(cam) #header, 
            # conn.send(application_id, header)
            conn.send(application_id, data) 
            print("send over")
            buffer_num += 1


def get_img_bytes(cam):
    img = cam.get_frame()
    img = Resize(img, 400)
    print(img.shape)
    return img2bytearray(img)

class TmpCam:
    def __init__(self, file_path):
        self.file_path = file_path
        self.files = os.listdir(self.file_path)
        self.ind = 0
        self.total_num = len(self.files)

    def get_frame(self):
        img = cv.imread(os.path.join(self.file_path, self.files[self.ind]))
        self.ind = (self.ind + 1) % self.total_num
        return img


if __name__ == "__main__":
    # client_run()
    pass





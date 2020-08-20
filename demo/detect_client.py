from detection import Camera, Resize
from .comm import ClientConnection
import pickle, sys, struct
from time import sleep
import numpy as np
import cv2 as cv


##TODO while first or connectiong firsts
def client_run(sleep_time=100):
    cam = Camera()
    IP = "127.0.0.1"
    Port = 10080
    application_id = 7
    with ClientConnection(IP, Port) as conn:
        print("detection client connected")
        while True:
            data = get_img_bytes(cam) #header, 
            # conn.send(application_id, header)
            conn.send(application_id, data) 
            print("send over")
            if cv.waitKey(20)==27:
                break

def get_img_bytes(cam):
    img = cam.get_frame() #cv.imread("data/4.png") #
    img = Resize(img, 400)
    shape = bytearray(np.array(img.shape))
    data = shape + bytearray(img)
    # size = sys.getsizeof(data)
    # header = struct.pack("i", size)
    return data #header, 

if __name__ == "__main__":
    client_run()





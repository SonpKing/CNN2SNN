from .comm import ServerListener
import pickle, sys, struct
from time import sleep
import cv2 as cv
import numpy as np


def server_run():
    IP = "localhost"
    Port = 10080
    with ServerListener(IP, Port, 100, 5) as listener:
        print("detection server listening")
        with listener.accept() as conn:
            print("detection server connected")
            while True:
                try:
                    data = None
                    while data == None:
                        _, data = conn.receive()
                        # sleep(0.1)
                    print("recevied")
                    shape = np.frombuffer(data[:12], dtype=np.int32)
                    data = np.frombuffer(data[12:], dtype=np.uint8)
                    data = data.reshape(shape)
                    cv.imshow('frame', data)
                    if cv.waitKey(20)==27:
                        break
                except ServerListener.TimeoutException:
                    pass
                except:
                    print('An exception is raised when processing the client request')
                

if __name__ == "__main__":
    server_run()
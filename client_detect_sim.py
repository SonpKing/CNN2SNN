# import sys
# sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
from detection import SelectiveSearch, Camera, generate_boxes, generate_bb, show_bb, vis_bb, Resize, filter_rects
from detection.generate_img_from_video import generate_img
import cv2 as cv
from models.BananaNet import banananetv2_spike
from models.MobileNet_Slim import mobilenet_slim_spike
from models import SpikeNet
from util import load_pretrained
from util.validate import eval_single
import torch
import numpy as np
import pickle
import cv2 as cv


from demo.detect_client import client_run
from demo.detect_server import server_run
from multiprocessing import Process


if __name__ == "__main__":
    # server_run("localhost", 10080)
    # client_run("localhost", 10080)
    client_run("192.168.1.202", 10080)
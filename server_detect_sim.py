# -*- coding: utf-8 -*-
"""
Created on Fri Jul  3 17:15:31 2020
@author: Paul King
"""
from detection import Camera, generate_boxes, generate_bb, show_bb, vis_bb, Resize, filter_rects, SelectiveSearch
from detection.generate_img_from_video import generate_img
from models.BananaNet import banananetv2_spike
from models.MobileNet_Slim import mobilenet_slim_spike
from models import SpikeNet
from util import load_pretrained
from util.validate import eval_single
from demo.detect_client import client_run
from demo.detect_server import server_run, test_server_run
from multiprocessing import Process
import torch
import numpy as np
import pickle
import cv2 as cv

if __name__ == "__main__":
    # server_run("localhost", 10080)
    server_run("192.168.1.202", 10080)
    # test_server_run("192.168.1.202", 10080)


'''
    put one img
    spike_time:  0.5026555061340332
    infer over [3 4 5] 192.168.1.6
    Total Number of Region Proposals: 293 , time cost 0.2772388458251953
    spike_time:  0.5026557445526123
    infer over [24 25 26] 192.168.1.17
    spike_time:  0.5036847591400146
    infer over [30 31 32] 192.168.1.19
    spike_time:  0.5036520957946777
    infer over [12 13 14] 192.168.1.13
    spike_time:  0.5026538372039795
    infer over [21 22 23] 192.168.1.16
    spike_time:  0.502655029296875
    infer over [18 19 20] 192.168.1.15
    spike_time:  0.502655029296875
    spike_time:  0.5036523342132568
    infer over [ 9 10 11] 192.168.1.11
    infer over [15 16 17] 192.168.1.14
    spike_time:  0.5046775341033936
    infer over [27 28 29] 192.168.1.18



    ----------------
    recevied

infer over [27 28 29] 192.168.1.50
infer over [45 46 47] 192.168.1.56
infer over [42 43 44] 192.168.1.55
infer over [24 25 26] 192.168.1.49
infer over [57 58 59] 192.168.1.60
infer over [39 40 41] 192.168.1.54
infer over [21 22 23] 192.168.1.48
infer over [15 16 17] 192.168.1.46
infer over [51 52 53] 192.168.1.58
infer over [48 49 50] 192.168.1.57
infer over [54 55 56] 192.168.1.59
infer over [12 13 14] 192.168.1.45
infer over [6 7 8] 192.168.1.43
infer over [0 1 2] 192.168.1.41
infer over [18 19 20] 192.168.1.47
infer over [33 34 35] 192.168.1.52
infer over [ 9 10 11] 192.168.1.44
infer over [3 4 5] 192.168.1.42
infer over [30 31 32] 192.168.1.51


------
spike_time:  0.5006601810455322
infer over [24 25 26] 192.168.1.49
Total Number of Region Proposals: 393 , time cost 0.34407901763916016
get input for searcher 1
spike_time:  0.5006604194641113
infer over [48 49 50] 192.168.1.57
spike_time:  0.5016579627990723
infer over [45 46 47] 192.168.1.56
spike_time:  0.5016577243804932
infer over [0 1 2] 192.168.1.41
spike_time:  0.5016574859619141
infer over [33 34 35] 192.168.1.52
recevied
spike_time:  0.502655029296875
infer over [57 58 59] 192.168.1.60
spike_time:  0.5016870498657227
infer over [51 52 53] 192.168.1.58
spike_time:  0.5027060508728027
infer over [6 7 8] 192.168.1.43
spike_time:  0.5006880760192871
infer over [39 40 41] 192.168.1.54
spike_time:  0.5006906986236572
infer over [42 43 44] 192.168.1.55
spike_time:  0.501657247543335
infer over [30 31 32] 192.168.1.51
spike_time:  0.5016577243804932
infer over [12 13 14] 192.168.1.45
spike_time:  0.5016570091247559
infer over [54 55 56] 192.168.1.59
spike_time:  0.5026555061340332
infer over [3 4 5] 192.168.1.42
spike_time:  0.503652811050415
infer over [21 22 23] 192.168.1.48


-------------48, 52, 53

spike_time:  0.5036537647247314
infer over [24 25 26] 192.168.1.49
recevied
spike_time:  0.5046510696411133
infer over [30 31 32] 192.168.1.51
spike_time:  0.5056765079498291
infer over [48 49 50] 192.168.1.57
recevied
spike_time:  0.5086398124694824
infer over [3 4 5] 192.168.1.42
spike_time:  0.5096375942230225
infer over [51 52 53] 192.168.1.58
spike_time:  0.5086400508880615
infer over [15 16 17] 192.168.1.46
spike_time:  0.5106353759765625
infer over [57 58 59] 192.168.1.60
spike_time:  0.5086402893066406
spike_time:  0.5086402893066406
infer over [42 43 44] 192.168.1.55
infer over [45 46 47] 192.168.1.56
spike_time:  0.5096731185913086
infer over [33 34 35] 192.168.1.52
spike_time:  0.5086703300476074
infer over [18 19 20] 192.168.1.47
spike_time:  0.5096361637115479
infer over [54 55 56] 192.168.1.59
'''
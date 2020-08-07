from detection import SelectiveSearch, Camera, generate_boxes, generate_bb, show_bb, vis_bb, Resize, filter_rects
from detection.generate_img_from_video import generate_img
import cv2 as cv
from models.AppleNet import applenetv2_spike
from models import SpikeNet
from util import load_pretrained
from util.validate import eval_single
import torch
import numpy as np
import pickle
import cv2 as cv


generate_img("data\WIN_20200805_14_55_40_Pro.mp4", "C:\\Users\\dell\\Pictures", 250, seed="new3")
# cam = Camera()
# # cam.display_video()
# img = cam.get_frame()
# img = cam.get_frame()
# img = cam.get_frame()

# img = cv.imread("data/2.png")
# #using mean and std to filter background
# mean, std = cv.meanStdDev(img)
# print(mean, std)
# img = Resize(img, 400)
# searcher = SelectiveSearch()
# rects = searcher.select(img)
# print("bound boxes generation done")
# boxes = filter_rects(img, rects)
# show_bb(img, boxes)

# model = applenetv2_spike(8, vth=90.0)
# pretrain = "checkpoint/0/applenet_normalise_scale.pth.tar"
# load_pretrained(model, pretrain, [], device=torch.device("cpu"))
# model = SpikeNet(model, vth=90.0)
# print("load the model done")

# ticks = 300
# cls_pred = []
# for box in boxes:
#     img_bb = generate_bb(img, box)
#     cv.imshow("Output", img_bb)
#     # mean, std = cv.meanStdDev(img_bb)
#     # print(mean, std)
#     cv.waitKey(1000)
#     img_bb = [img_bb.transpose((2, 0, 1))]
#     inputs = torch.tensor(img_bb)
#     result = eval_single(inputs, model, ticks)
#     print(result)
#     cls_pred.append(result.numpy()[0])
# cv.destroyAllWindows()

# with open("img.tmp", "wb") as f:
#     pickle.dump(img, f)
# with open("boxes.tmp", "wb") as f:
#     pickle.dump(boxes, f)
# with open("cls_pred.tmp", "wb") as f:
#     pickle.dump(cls_pred, f)

# with open("img.tmp", "rb") as f:
#     img = pickle.load(f)
# with open("boxes.tmp", "rb") as f:
#     boxes = pickle.load(f)
# with open("cls_pred.tmp", "rb") as f:
#     cls_pred = pickle.load(f)

# boxes, cls_inds, scores = generate_boxes(boxes, cls_pred, cls_thresh=0.955)
# print(boxes)
# print(cls_inds)
# print(scores)
# class_name = ['Aeroplane', 'Apple', 'Banana', 'Car', 'Horse', 'Keyboard', 'Mug', 'Poodle']
# vis_bb(img, boxes, scores, cls_inds, 8, class_name)

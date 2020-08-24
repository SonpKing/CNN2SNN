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

    # p1=Process(target=server_run, args=("localhost", 10080,)) #必须加,号 cxcxcx
    # p2=Process(target=client_run, args=("localhost", 10080,))
    # p1.start()
    # p2.start()
    # p1.join()
    # p2.join()



    # sgenerate_img("C:\\Users\\dell\\Desktop\\videos\\video3.mov", "C:\\Users\\dell\\Pictures\\Saved Pictures", 200, seed="new6")
# # cam = Camera()
# # # cam.display_video()
# # img = cam.get_frame()
# # img = cam.get_frame()
# # img = cam.get_frame()

    # img = cv.imread("data/4.png")
    # #using mean and std to filter background
    # mean, std = cv.meanStdDev(img)
    # print(mean, std)
    # img = Resize(img, 400)
    # searcher = SelectiveSearch()
    # rects = searcher.select(img)
    # print("bound boxes generation done")
    # boxes = filter_rects(img, rects)
    # show_bb(img, boxes)

    # # model = mobilenet_slim_spike(7, vth=70.0)
    # model = banananetv2_spike(7, 100)
    # pretrain = "checkpoint\\0\\banananet_normalise_scale100.pth.tar"
    # load_pretrained(model, pretrain, [], device=torch.device("cpu"))
    # model = SpikeNet(model, vth=70.0)
    # print("load the model done")

    # ticks = 200
    # cls_pred = []
    # for box in boxes:
    #     img_bb = generate_bb(img, box)
    #     # mean, std = cv.meanStdDev(img_bb)
    #     # print(mean, std)
    #     # cv.waitKey(1000)
    #     img_bb = [img_bb.transpose((2, 0, 1))]
    #     inputs = torch.tensor(img_bb)
    #     result = eval_single(inputs, model, ticks)
    #     print(result)
    #     cls_pred.append(result.numpy()[0])
    # cv.destroyAllWindows()

    # with open("img1.tmp", "wb") as f:
    #     pickle.dump(img, f)
    # with open("boxes1.tmp", "wb") as f:
    #     pickle.dump(boxes, f)
    # with open("cls_pred1.tmp", "wb") as f:
    #     pickle.dump(cls_pred, f)

    # with open("img.tmp", "rb") as f:
    #     img = pickle.load(f)
    # with open("boxes.tmp", "rb") as f:
    #     boxes = pickle.load(f)
    # with open("cls_pred.tmp", "rb") as f:
    #     cls_pred = pickle.load(f)

    # boxes, cls_inds, scores = generate_boxes(boxes, cls_pred, cls_thresh=0.955, nms_thresh=0.5, scores_rm=[])
    # print(boxes)
    # print(cls_inds)
    # print(scores)
    # class_name = ['broker', 'diba', 'floor', 'house', 'person', 'shoe', 'water']
    # vis_bb(img, boxes, scores, cls_inds, 7, class_name)


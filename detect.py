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
from util import data_loader


from demo.detect_client import client_run
from demo.detect_server import server_run
from multiprocessing import Process

def main():
    _, val_loader = data_loader("C:\\Users\\dell\\Desktop\\Detect_Data", batch_size=1, img_size=32, workers=1, dataset="imagenet")
    
    # model = mobilenet_slim_spike(14,_if=False, vth=70.0)
    # pretrain = "checkpoint\\0\\slim_cls14_normalise_70.pth.tar"
    # load_pretrained(model, pretrain, [], device=torch.device("cpu"))
    # total_acc  = 0
    # for it, (inputs, targets) in enumerate(val_loader):
    #     sim_res = eval_single(inputs, model, 1)
    #     sim_res = np.argmax(sim_res)
    #     if sim_res == targets[0]:
    #         total_acc += 1
    #     print(it,sim_res, targets[0], total_acc/(it+1)*100)



if __name__ == "__main__":
    # main()
    # exit(0)
    # server_run("localhost", 10080)
    # client_run("localhost", 10080)
    # p1=Process(target=server_run, args=("localhost", 10080,)) #必须加,号 cxcxcx
    # p2=Process(target=client_run, args=("localhost", 10080,))
    # p1.start()
    # p2.start()
    # p1.join()
    # p2.join()



    # generate_img("C:\\Users\\dell\\Desktop\\videos\\video4.mov", "C:\\Users\\dell\\Pictures\\Camera Roll", 77, seed="new3")
# # cam = Camera()
# # # cam.display_video()
# # img = cam.get_frame()
# # img = cam.get_frame()
# # img = cam.get_frame()

    from demo.detect_client import TmpCam
    cam = TmpCam("C:\\Users\\dell\\Desktop\\Detect_Data\\mytest")
    model = mobilenet_slim_spike(14,_if=False, vth=70.0)
    pretrain = "checkpoint\\0\\slim_cls14_limitcrop_normalise_70.pth.tar"
    load_pretrained(model, pretrain, [], device=torch.device("cpu"))
    # model = SpikeNet(model, vth=50.0)
    ticks = 1
    print("load the model done")

    while True:
        img = cam.get_frame()
        #using mean and std to filter background
        # mean, std = cv.meanStdDev(img)
        # print(mean, std)
        img = Resize(img, 400)
        searcher = SelectiveSearch(quality=False)
        rects = searcher.select(img)                                              

        print("bound boxes generation done")
        boxes = filter_rects(img, rects, size=60)
        show_bb(img, boxes)
        cv.destroyAllWindows()
        inputs = []
        for box in boxes:
            img_bb = generate_bb(img, box)
            # cv.imshow( "Output", img_bb)
            # cv.waitKey(1000)
            # mean, std = cv.meanStdDev(img_bb)
            # print(mean, std)
            # cv.waitKey(1000)
            inputs.append(img_bb.transpose((2, 0, 1)))
        inputs = torch.tensor(inputs).float()
        cls_pred = eval_single(inputs, model, ticks)
        cls_pred = [item[8:] for item in cls_pred.numpy()]
        print(cls_pred)

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
        # scores_rm = [0, 1, 2, 3, 4, 5, 6, 7]
        boxes, cls_inds, scores = generate_boxes(boxes, cls_pred, cls_thresh=0.1, nms_thresh=0.3, scores_rm=[2])
        print(boxes)
        print(cls_inds)
        print(scores)
        # class_name = ['Aeroplane', 'Apple', 'Banana', 'Car', 'Horse', 'Keyboard', 'Mug', 'Poodle', 'broker', 'diba', 'floor', 'house', 'person', 'water']
        class_name = ['broker', 'diba', 'floor', 'water', 'person', 'house']
        vis_bb(img, boxes, scores, cls_inds, class_name, show_time=5000, show=True)


import pickle, sys, struct
from time import sleep
import cv2 as cv
import numpy as np
from detection import SelectiveSearch, Camera, generate_boxes, generate_bb, show_bb, vis_bb, Resize, filter_rects
from detection.generate_img_from_video import generate_img
import math
from .detect_util import *
from util import load_pretrained
from util.validate import eval_single
from models import SpikeNet, to_tensor
from models.MobileNet_Slim import mobilenet_slim_spike
from .comm import ServerListener
from .visual.web_visual import show_async
from .l_short_comm import Get
from multiprocessing import Manager
import torch
from hardware.darwain_hardware import DarwinDev, generate_multi_input, fit_input
from util.util import Timer
import time
    

def start_service(IP, Port, conns, thread_num, res_que, plt_que, ctl_que):
    recv_pool = PoolHelper(thread_num)
    searcher = SelectiveSearch(quality=-1)
    application_id = 7
    timer = Timer()
    total_num = thread_num * 3
    last_robot_cmd = -1
    last_rotate_time = time.time()
    last_robot_time = time.time()
    with ServerListener(IP, Port, None, 5) as listener:
        print("detection server listening")
        with listener.accept() as conn:
            print("detection server connected")
            while True:
                try:
                    data = recv_until_ok(conn)
                    img = bytearray2img(data)
                    timer.record("starting loop")
                    package_inputs, boxes = generate_input(searcher, img, total_num)
                    timer.record("generate input over")
                    distribute(conns, package_inputs, application_id)
                    timer.record("distribute input to inferencer over")
                    res = collect_output(conns, recv_pool, res_que, total_num)
                    # new_conns = []
                    # for i in res:
                    #     if i%3 == 0:
                    #         new_conns.append(conns[i//3])
                    # conns = new_conns
                    # total_num = len(conns)
                    timer.record("collect output over")
                    
                    is_water, is_house, is_broker, is_person = process_output(img, res, boxes, plt_que)
                    print(is_water, is_house, is_broker, is_person)
                    try:
                        robot_IP = "192.168.2.100"
                        robot_Port = 13001
                        cur_time = time.time()
                        if cur_time - last_robot_time > 10:
                            last_robot_cmd = -1
                        if cur_time - last_rotate_time < 30:
                            can_rotate = False
                        else:
                            can_rotate = True
                        if is_broker or is_house or is_water or is_person:
                            if is_person and last_robot_cmd != 0:
                                Get(robot_IP, robot_Port, 100, bytearray(2))
                                Get(robot_IP, robot_Port, 104, bytearray(2))
                                print("!!!!!!!!!!!!!!!send person", last_robot_cmd)
                                last_robot_cmd = 0
                                last_robot_time = cur_time
                            elif is_water and last_robot_cmd != 1:
                                Get(robot_IP, robot_Port, 102, bytearray(2))
                                Get(robot_IP, robot_Port, 103, bytearray(2))
                                Get(robot_IP, robot_Port, 104, bytearray(2))
                                print("!!!!!!!!!!!!!!!send broker", last_robot_cmd)
                                last_robot_cmd = 1
                                last_robot_time = cur_time
                            elif (is_house or is_broker) and last_robot_cmd != 2 and can_rotate:
                                # Get(robot_IP, robot_Port, 105, bytearray(2))
                                # print("!!!!!!!!!!!!!!!send rotate", last_robot_cmd)
                                last_robot_cmd = 2
                                last_rotate_time = cur_time
                                last_robot_time = cur_time

                    except:
                        pass
                    timer.record("process output over")
                    conn.send(application_id, bytearray(1))
                except Exception as e:
                    print('An exception is raised when processing the client request', e)
                    # recv_pool.close()
                    raise(e)
                if not ctl_que.empty():
                    # recv_pool.close()
                    break
    # recv_pool.close()

def generate_input(searcher, img, boundbox_num=48, package_num=3):
    rects = searcher.select(img)
    boundbox_num = math.ceil(boundbox_num // package_num) * package_num
    boxes = filter_rects(img, rects, size=boundbox_num)
    img_bbs = []
    for i, box in enumerate(boxes):
        img_bbs.append(generate_bb(img, box))
    img_bbs = np.array(img_bbs)
    indexs = np.arange(len(boxes))

    package_inputs = []
    for i in range(len(boxes)//package_num):
        ind = i*package_num
        package_inputs.append((indexs[ind: ind+package_num], img_bbs[ind: ind+package_num]))
    return package_inputs, boxes


def distribute(conns, package_inputs, application_id=7):
    for conn in conns:
        assert conn is not None
    for i in range(len(package_inputs)):
        conns[i].send(package_to_bytearray(package_inputs[i]))

def collect_output(conns, recv_pool, res_que, total_num):
    for conn in conns:
        recv_pool.execute(receive_async, (conn, res_que,))
    # max_try_time = 100
    # cur_try_time = 0
    last_time = time.time()
    res = dict()
    while len(res) < total_num:
        if not res_que.empty():
            while not res_que.empty():
                bytes_data = res_que.get()
                if not isinstance(bytes_data, bytearray):
                    print("exit collect output")
                    return None
                ind, outs = bytearray_to_class_out(bytes_data)
                for i in range(len(ind)):
                    res[ind[i]] = outs[i]
                # print("currently", len(res), "result collected")
        else:
            sleep(0.1)
            # cur_try_time += 1
            # if cur_try_time > max_try_time:
            #     return res  
            if(time.time() - last_time > 2):
                break
    # print(res)
    return res

def process_output(img, res, boxes, plt_que):
    is_water = is_broker = is_house = is_person = False
    if res is None:
        print("ignore this frame")
    else:
        new_boxes = []
        cls_pred = []
        for i in range(len(boxes)):
            if i in res:
                new_boxes.append(boxes[i])
                cls_pred.append(res[i])
        print("receive totally", len(new_boxes), "results")
        boxes, cls_inds, scores = generate_boxes(boxes, cls_pred, cls_thresh=0.65, nms_thresh=0.4, scores_rm=[])
        # boxes, cls_inds, scores = generate_boxes(boxes, cls_pred, cls_thresh=0.01, nms_thresh=0.9, scores_rm=[2])
        # print(boxes)
        print(cls_inds)
        print(scores)
        class_name = ['broker', 'diba', 'floor', 'water', 'person', 'house']
        fig = vis_bb(img, boxes, scores, cls_inds, class_name, show=True)#, show_time=5000, show=True
        plt_que.put(fig)
        for cls_ind in cls_inds:
            if cls_ind == 6:
                is_water = True
            if cls_ind == 4:
                is_person = True
            if cls_ind == 3:
                is_house = True
            if cls_ind == 0:
                is_broker = True
    return is_water, is_house, is_broker, is_person

def receive_async(conn, res_que):
    try:
        res_que.put(conn.recv())
        # print("get output")
    except:
        print("queue is full")
        exit(0)
       
class AbsConnection:
    def __init__(self, in_que, res_que):
        self.in_que = in_que
        self.res_que = res_que

    def send(self, args):
        self.in_que.put(args)
        
    def recv(self):
        while True:
            if not self.res_que.empty():
                return self.res_que.get()
            else:
                sleep(0.01)

    def close(self):
        self.in_que.put("exit")

def inference_async(pretrained_path, vth, in_que, res_que):
    print("start inference instance")
    model = mobilenet_slim_spike(14,_if=False, vth=70.0)
    load_pretrained(model, pretrained_path, [], device=torch.device("cpu"))
    # print("ok")
    # model = SpikeNet(model, vth=vth)
    print("loaded model in pool process")
    while True:
        if in_que.empty():
            sleep(0.01)
        else:
            bytes_data = in_que.get()
            if not isinstance(bytes_data, bytearray):
                print("exit inference_async")
                res_que.put("exit")
                break
            inds, imgs = bytearray_to_package(bytes_data)
            inputs = to_tensor(fit_input_batch(imgs))
            print("start infering")
            result = eval_single(inputs, model, 1)
            print("infer over", inds)
            res_que.put(class_out_to_bytearray(inds, result[:, 8:]))

class FalseConnection(AbsConnection):
    def __init__(self, in_que, res_que, helper, pretrained_path, vth):
        super().__init__(in_que, res_que)
        helper.execute(inference_async, (pretrained_path, vth, self.in_que, self.res_que,))

def inference_dev(IP, Port, class_num, in_que, res_que):
    print("starting init dev")
    dev = DarwinDev(IP, Port, 220000, class_num, False) #####!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    while True:
        if in_que.empty():
            sleep(0.01)
        else:
            bytes_data = in_que.get()
            if not isinstance(bytes_data, bytearray):
                print("exit inference_async")
                res_que.put("exit")
                break
            inds, imgs = bytearray_to_package(bytes_data)
            img0 = fit_input(imgs[0])
            img1 = fit_input(imgs[1])
            img2 = fit_input(imgs[2])
            inputlist, rowlist = generate_multi_input(img0, img1, img2)
            print("start infering", inds, IP)
            dev.eliminate('all')
            dev.run(inputlist, rowlist, 200)
            result = dev.get_result()
            result = np.array(result).reshape((3,-1))
            print("infer over", inds, IP)
            res_que.put(class_out_to_bytearray(inds, result[:, 8:]))


class DarwinConnection(AbsConnection):
    def __init__(self, IP, Port, class_num, in_que, res_que, helper):
        super().__init__(in_que, res_que)
        helper.execute(inference_dev, (IP, Port, class_num, self.in_que, self.res_que,))
        
# from demo.visual.WebDraw import WebDraw
# import matplotlib.pyplot as plt
# from random import randint
# from time import sleep

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
#                 sleep(0.01)
#             fig = plt_que.get()
#             draw(fig)
#         except:
#             continue

def server_run(IP, Port):
    IPs = [2, 3, 6, 7, 11, 12, 13, 14, 15, 16, 17, 18, 19, 21]#, 20, 8
    thread_num = len(IPs)
    inference_pool = PoolHelper(thread_num)
    conns = []
    pretrained_path = "checkpoint\\0\\slim_nice_normalise_70.pth.tar"
    vth = 70
    manager = Manager()
    class_num = 14
    for i in range(thread_num):
        conns.append(FalseConnection(manager.Queue(), manager.Queue(), inference_pool, pretrained_path, vth))
        # conns.append(DarwinConnection("192.168.1."+str(IPs[i]), 7, class_num, manager.Queue(), manager.Queue(), inference_pool))
    res_que = manager.Queue()

    plt_que = manager.Queue()

    ctl_que = manager.Queue()
    try:
        master_process = Process(target=start_service, args=(IP, Port, conns, thread_num, res_que, plt_que, ctl_que,)) 
        visual_process = Process(target=show_async, args=(plt_que, ctl_que,))  
        master_process.start()
        visual_process.start()
        master_process.join()
        visual_process.join()
    finally:
        for sub_conn in conns:
            sub_conn.close()
        ctl_que.put("close")
        exit(0)

if __name__ == "__main__":
    server_run("localhost", 10080)




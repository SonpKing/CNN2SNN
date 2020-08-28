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
from multiprocessing import Manager
import torch
from hardware.darwain_hardware import DarwinDev, generate_multi_input, fit_input
from util.util import Timer
import threading
import queue
from .robot import Robot
import time
    
SLEEP_TIME = 0.01
RECV_TIME = 0.25
REVIEW_TIME = 0.15
WAIT_TIME = 2

def start_service(conns, thread_num, input_que, res_que, plt_que, ctl_que, seed_que):
    recv_pool = PoolHelper(thread_num)
    timer = Timer()
    robot = Robot()
    img_id = 1
    class_name = ['broker', 'diba', 'floor', 'water', 'person', 'robot', 'house']
    class_color = [(85, 115, 139),
                    (34, 139, 34),
                    (201, 201, 201),
                    (255, 144, 30),
                    (48, 48, 255),
                    (205, 50, 154),
                    (0, 102, 205)]
    # class_color = [(np.random.randint(255),np.random.randint(255),np.random.randint(255)) for _ in range(len(class_name))]
    has_water = False
    # scores_rm = []
    scores_rm = [4]
    while True:
        timer.record("starting loop", img_id)
        package_inputs, boxes, img = generate_input(seed_que, input_que, thread_num, img_id)
        timer.record("generate input over", img_id)
        if len(boxes) <= 0:
            timer.record("no boxes", img_id)
            continue
        distribute(conns, package_inputs)
        timer.record("distribute", len(boxes), "to inferencer over", img_id)
        res = collect_output(conns, recv_pool, res_que, len(boxes), img_id)
        timer.record("collect output over", img_id)
        is_water, is_house, is_broker, is_person = process_output(img, res, boxes, plt_que, class_name, class_color, scores_rm = scores_rm)
        if has_water:
            scores_rm = [3]
            
        else:
            has_water = is_water
            scores_rm = [4]
        timer.record("process output over", img_id)
        robot.control(is_water=is_water, is_house=False, is_broker = False, is_person=is_person)
        if not ctl_que.empty():
            # recv_pool.close()
            break
        img_id = (img_id + 1) % 1133579


def recv_img_async(IP, Port, input_que, plt_que, seed_que):
    application_id = 7
    with ServerListener(IP, Port, None, 5) as listener:
        print("detection server listening")
        with listener.accept() as conn:
            print("detection server connected")
            img_que = queue.Queue()
            tmp_que = queue.Queue()
            searcher_thread = threading.Thread(target=searcher_input, args=(img_que, input_que, tmp_que))
            send_thread = threading.Thread(target=send_real_time_image, args=(tmp_que, plt_que, seed_que))
            searcher_thread.start()
            send_thread.start()
            while True:
                try:
                    data = recv_until_ok(conn)
                    img = bytearray2img(data)
                    img_que.put(img)
                    conn.send(application_id, bytearray(1))
                    sleep(RECV_TIME)
                except Exception as e:
                    print('An exception is raised when processing the client request', e)
                    raise(e)


def searcher_input(img_que, input_que, tmp_que):
    print("start searcher thread")
    searcher = SelectiveSearch(quality=-1)
    qlen = img_que.qsize()
    private_que = queue.Queue()
    for _ in range(qlen - 1):
        private_que.put(img_que.get())
    seed = 0
    while True:
        if not img_que.empty():
            qlen = img_que.qsize()
            print("get input for searcher", qlen)
            for _ in range(qlen - 1):
                private_que.put(img_que.get())
            img = img_que.get()
            private_que.put(img)
            rects = searcher.select(img)
            input_que.put((img, rects, seed))
            while not private_que.empty():
                tmp_que.put((seed, private_que.get()))
            seed = (seed + 1) % 1133579
        else:
            sleep(SLEEP_TIME)

def send_real_time_image(tmp_que, plt_que, seed_que):
    cur_seed = -1
    last_img = None
    # last_time = time.time()
    while True:
        while seed_que.empty():
            sleep(SLEEP_TIME)
        cur_seed = seed_que.get()
        if last_img is not None:
            plt_que.put(last_img)
            last_img = None
        # cur_time = time.time()
        # dt_time = (cur_time - last_time)/tmp_que.qsize()
        # last_time = cur_time
        while not tmp_que.empty():
            seed, img = tmp_que.get()
            if seed <= cur_seed:
                plt_que.put(img)
                print("put one img")
                sleep(REVIEW_TIME)
            else:
                last_img = img
                break
        sleep(SLEEP_TIME)


def generate_input(seed_que, input_que, thread_num, img_id, package_num=3):
    while input_que.empty():
        sleep(SLEEP_TIME)
    qlen = input_que.qsize()
    for _ in range(qlen - 1):
        input_que.get()
    img, rects, seed = input_que.get()
    seed_que.put(seed)
    boundbox_num = thread_num * package_num
    boxes = filter_rects(img, rects, size=boundbox_num)
    img_bbs = []
    for i, box in enumerate(boxes):
        img_bbs.append(generate_bb(img, box))
    img_bbs = np.array(img_bbs)
    indexs = np.arange(len(boxes))
    package_inputs = []
    for i in range(len(boxes)//package_num):
        ind = i*package_num
        package_inputs.append((indexs[ind: ind+package_num], img_bbs[ind: ind+package_num], [img_id]*package_num))
    return package_inputs, boxes, img
    

def distribute(conns, package_inputs, application_id=7):
    for conn in conns:
        assert conn is not None
    for i in range(len(package_inputs)):
        conns[i].send(package_to_bytearray(package_inputs[i]))

def collect_output(conns, recv_pool, res_que, total_num, img_id):
    last_time = time.time()
    res = dict()
    while len(res) < total_num:
        if not res_que.empty():
            while not res_que.empty():
                tuple_data = res_que.get()
                if not isinstance(tuple_data, tuple):
                    print("exit collect output")
                    return None
                ind, outs, img_ids = tuple_data
                for i in range(len(ind)):
                    if img_ids[i] == img_id:
                        res[ind[i]] = outs[i]
                # print("currently", len(res), "result collected")
        else:
            sleep(SLEEP_TIME)
            if(time.time() - last_time > WAIT_TIME):
                break
    # print(res)
    return res

def process_output(img, res, boxes, plt_que, class_name, class_color, scores_rm=[]):
    is_water = is_broker = is_house = is_person = False
    if res is None or len(res) == 0:
        print("ignore this frame")
    else:
        new_boxes = []
        cls_pred = []
        for i in range(len(boxes)):
            if i in res:
                new_boxes.append(boxes[i])
                cls_pred.append(res[i])
        print("receive totally", len(new_boxes), "results")
        boxes, cls_inds, scores = generate_boxes(boxes, cls_pred, cls_thresh=0.3, nms_thresh=0.5, scores_rm=scores_rm)
        # boxes, cls_inds, scores = generate_boxes(boxes, cls_pred, cls_thresh=0.01, nms_thresh=0.9, scores_rm=[2])
        # print(boxes)
        print(cls_inds)
        print(scores)
        
        
        fig = vis_bb(img, boxes, scores, cls_inds, class_name, class_color)#, show_time=5000, show=True
        plt_que.put(fig)
        plt_que.put(fig)
        plt_que.put(fig)
        print("put cls img")
        for cls_ind in cls_inds:
            if cls_ind == 6:
                is_house = True
            if cls_ind == 4:
                is_person = True
            if cls_ind == 3:
                is_water = True
            if cls_ind == 0:
                is_broker = True
    return is_water, is_house, is_broker, is_person

# def receive_async(conn, res_que):
#     try:
#         res_que.put(conn.recv())
#         # print("get output")
#     except:
#         print("queue is full")
#         exit(0)
       
class AbsConnection:
    def __init__(self, in_que, res_que):
        self.in_que = in_que
        self.res_que = res_que

    def send(self, args):
        while not self.in_que.empty():
            _ = self.in_que.get()
        self.in_que.put(args)
        
    # def recv(self):
    #     while True:
    #         if not self.res_que.empty():
    #             return self.res_que.get()
    #         else:
    #             sleep(SLEEP_TIME)

    def close(self):
        self.in_que.put("exit")

def inference_async(pretrained_path, vth, in_que, res_que):
    print("start inference instance")
    # vth = pretrained_path.split("\\")[-1].split(".")[0][-2:]
    # print("vth", vth)
    model = mobilenet_slim_spike(14,_if=False, vth=vth)
    load_pretrained(model, pretrained_path, [], device=torch.device("cpu"))
    # print("ok")
    # model = SpikeNet(model, vth=vth)
    print("loaded model in pool process")
    while True:
        if in_que.empty():
            sleep(SLEEP_TIME)
        else:
            bytes_data = in_que.get()
            if not isinstance(bytes_data, bytearray):
                print("exit inference_async")
                res_que.put("exit")
                break
            inds, imgs, img_ids = bytearray_to_package(bytes_data)
            inputs = to_tensor(fit_input_batch(imgs))
            # print("start infering")
            result = np.array(eval_single(inputs, model, 1))
            print("infer over", inds)
            # sleep(1)
            res_que.put((inds, result[:, 7:], img_ids))

class FalseConnection(AbsConnection):
    def __init__(self, in_que, res_que, helper, pretrained_path, vth):
        super().__init__(in_que, res_que)
        helper.execute(inference_async, (pretrained_path, vth, self.in_que, self.res_que,))

def inference_dev(IP, Port, class_num, in_que, res_que):
    print("starting init dev")
    dev = DarwinDev(IP, Port, 220000, class_num, False) #####!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    while True:
        if in_que.empty():
            sleep(SLEEP_TIME)
        else:
            bytes_data = in_que.get()
            if not isinstance(bytes_data, bytearray):
                print("exit inference_async")
                res_que.put("exit")
                break
            inds, imgs, img_ids = bytearray_to_package(bytes_data)
            img0 = fit_input(imgs[0])
            img1 = fit_input(imgs[1])
            img2 = fit_input(imgs[2])
            inputlist, rowlist = generate_multi_input(img0, img1, img2)
            print("start infering", inds, IP)
            dev.eliminate('all')
            try:
                dev.run(inputlist, rowlist, 200)
                result = dev.get_result()
                result = np.array(result).reshape((3,-1))
                print("infer over", inds, IP)
                res_que.put((inds, result[:, 7:], img_ids))
            except:
                print("infer error")
                


class DarwinConnection(AbsConnection):
    def __init__(self, IP, Port, class_num, in_que, res_que, helper):
        super().__init__(in_que, res_que)
        helper.execute(inference_dev, (IP, Port, class_num, self.in_que, self.res_que,))
    

def server_run(IP, Port):
    IPs = [41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60]#[3, 6, 7, 11, 13, 14, 15, 16, 17, 18, 19, 21]#, 20, 8, 22, 23, 24, 25, 26, 27, 28, 2, 5348,52,
    thread_num = len(IPs) #20 #
    inference_pool = PoolHelper(thread_num)
    conns = []
    pretrained_path = "checkpoint\\0\\slim_nice7_2_normalise_scale60.pth.tar"
    vth = 60
    manager = Manager()
    class_num = 14

    res_que = manager.Queue()

    for i in range(thread_num):
        # conns.append(FalseConnection(manager.Queue(), res_que, inference_pool, pretrained_path, vth))
        conns.append(DarwinConnection("192.168.1."+str(IPs[i]), 7, class_num, manager.Queue(), res_que, inference_pool))
    
    plt_que = manager.Queue()

    ctl_que = manager.Queue()

    input_que = manager.Queue()

    seed_que = manager.Queue()
    try:
        master_process = Process(target=start_service, args=(conns, thread_num, input_que, res_que, plt_que, ctl_que, seed_que)) 
        input_process = Process(target=recv_img_async, args=(IP, Port, input_que, plt_que, seed_que, )) 
        visual_process = Process(target=show_async, args=(plt_que, ctl_que,))  
        master_process.start()
        input_process.start()
        visual_process.start()
        master_process.join()
        input_process.join()
        visual_process.join()
    except:
        for sub_conn in conns:
            sub_conn.close()
        ctl_que.put("close")
        exit(0)

if __name__ == "__main__":
    server_run("localhost", 10080)




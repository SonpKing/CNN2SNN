import numpy as np
# from .selective_search import show_bb
max_conf = 0
def softmax(X, axis=-1):
    dim_max = np.expand_dims(np.max(X, axis=axis), axis=axis)
    X -= dim_max
    X_exp = np.exp(X)
    s = X_exp / np.sum(X_exp, axis=axis, keepdims=True)
    return s


def cls_score(cls_pred, boxes):
    '''
    cls_pred may look like following
    [
        [0, 3, 7, 18, 0],
        [3, 2, 9, 29, 1],
        ...
    ]
    '''
    # boxes = np.array(boxes)
    # print(boxes.shape)
    # areas = (boxes[:, 2] - boxes[:, 0])* (boxes[:, 3] - boxes[:, 1])
    # ratio  = np.log10(areas / np.min(areas)) * 2 / 3
    all_class = softmax(cls_pred, 1)
    cls_inds = np.argmax(all_class, 1)
    scores = all_class[(np.arange(all_class.shape[0]), cls_inds)] #* ratio
    return scores, cls_inds

def nms(boxes, scores, nms_thresh=0.5):
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1) * (y2 - y1)
    order = (-scores).argsort() #descend

    keep = []
    while order.size > 0:
        
        i = order[0]
        maxx1 = np.maximum(x1[i], x1[order[1:]])
        maxy1 = np.maximum(y1[i], y1[order[1:]])
        minx2 = np.minimum(x2[i], x2[order[1:]])
        miny2 = np.minimum(y2[i], y2[order[1:]])

        width = np.maximum(1e-28, minx2 - maxx1)
        height = np.maximum(1e-28, miny2 - maxy1)
        inter = width * height

        iou = inter / np.min(([areas[i]] * len(order[1:]), areas[order[1:]]), 0) #inter / (areas[i] + areas[order[1:]] - inter) #
        keep.append(i)
        inds = np.where(iou <= nms_thresh)[0]
        order = order[inds + 1]
        

    return keep


def generate_boxes(rects, cls_pred, cls_thresh=0.5, nms_thresh=0.5, scores_rm=[], anno=True):
    if anno:
        cls_pred = np.array(cls_pred)
        conf_pred = cls_pred[:, -1]
        cls_pred = cls_pred[:, :-1]
    scores, cls_inds = cls_score(cls_pred, rects)
    for ind in scores_rm:
        scores[cls_inds==ind] = 0
    # global max_conf
    # max_conf = max(np.max(conf_pred), max_conf)
    # print(conf_pred, "#########################max_conf:", max_conf)
    # conf_pred = conf_pred / 16.0
    if anno:
        scores = scores * conf_pred
        mask = np.where((scores >= cls_thresh) & (conf_pred > 0.5))#
    else:
        mask = np.where(scores >= cls_thresh)
    print(scores)
    scores = scores[mask]
    cls_inds = cls_inds[mask]
    boxes = rects[mask]
    keep_inds = nms(boxes, scores, nms_thresh)
    return boxes[keep_inds], cls_inds[keep_inds], scores[keep_inds]

if __name__ == "__main__":
    data =[
        [0, 3, 7, 18, 0],
        [3, 2, 9, 29, 1],
        [2, 8, 10, 0, 3],
        [0, 0, 0, 0, 0]]
    data = np.array(data)[:, :5]
    # data = data /  np.expand_dims(np.max(data, -1), -1)
    # data = np.arange(24).reshape(2, 3, 4)
    # import torch
    # print(torch.softmax(torch.tensor(data, dtype=np.float), dim=-3, dtype=np.float))
    # print(softmax(data))
    print(cls_score(data))
    
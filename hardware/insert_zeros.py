import os
import pickle
import numpy as np

def insert_zeros(path, num, save_path):
    files = os.listdir(path)
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    for file in files:
        with open(os.path.join(path, file), "rb") as f:
            data = pickle.load(f)
        data = sorted(data, key=lambda x: x[0])
        tmp = []
        last_id = (0, 0)
        last_conn = (data[0][0], data[0][1], 0, data[0][3])
        for i, d in enumerate(data):
            if d[0] != last_id[0]:
                cnt = i - last_id[1]
                if cnt < num:
                    for _ in range(num - cnt):
                        tmp.append(last_conn)
                last_id = (d[0], i)
                last_conn = (d[0], d[1], 0, d[3])
            tmp.append(d)
        with open(os.path.join(save_path, file), "wb") as f:
            pickle.dump(np.array(tmp), f)
        




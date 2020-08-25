import time

class Timer:
    def __init__(self):
        self.last_time = time.time()

    def record(self, *args):
        cur_time = time.time()
        print(*args, ", time cost", cur_time - self.last_time)
        self.last_time = cur_time
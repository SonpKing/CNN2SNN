import threading
import queue
import time
event=threading.Event()

def put_something(_q: queue.Queue):
    time.sleep(10)
    _q.put("some thing")
    event.set()

def wait_something_put(_q: queue.Queue):
    print("wait something put")
    while True:
        if not _q.empty():
            print("get something:%s" % _q.get())
            event.clear()
        else:
            event.wait()
if __name__ == "__main__":
    _q = queue.Queue()
    _t = []
    _t.append(threading.Thread(target=put_something, args=(_q,)))
    _t.append(threading.Thread(target=wait_something_put, args=(_q,)))
    for _tt in _t:
        _tt.start()
    for _tt in _t:
        _tt.join()
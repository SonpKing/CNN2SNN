import mpld3 
from websocket_server import WebsocketServer
from threading import Thread
from time import sleep
import json
import mpld3
import matplotlib.pyplot as plt

class WebDraw:
    def __init__(self, PORT):
        self.PORT=PORT
        server = WebsocketServer(PORT)
        server.set_fn_new_client(self.new_client)
        server.set_fn_client_left(self.client_left)
        server.set_fn_message_received(self.message_received)
        self.server = server
        self.server_thread = Thread(target=server.run_forever)
        self.server_thread.start()
    def __call__(self, fig):
        python_obj = mpld3.fig_to_dict(fig)
        json_str = json.dumps(python_obj)
        if len(self.server.clients) > 0:
            self.server.send_message(self.server.clients[-1], json_str)
        plt.close(fig)
    def new_client(self, client, server):
        pass
    def client_left(self, client, server):
        pass
    def message_received(self, client, server, message):
        pass







from .comm import ServerListener
from .comm import ClientConnection

import sys
def RunServer(ip,port,callback,client=1,timeout=5):
    server=ServerListener(ip,port,timeout,client)
    print("server start")
    while(1):
        try:
            with server.accept() as conn:
                _id,_data=conn.receive()
                _sid,_sdata=callback(_id,_data)
                conn.send(_sid,_sdata)
        except ServerListener.TimeoutException:
            pass
        except :
            print("err: ", sys.exc_info()[0])
            pass
    pass

def Get(server_ip,port,id,data):
    client=ClientConnection(server_ip,port)
    client.send(id,data)
    _id,_data=client.receive()
    client.close()
    return _id,_data
  

'''
192.168.1.100:13001
id=100, robot2
id=102, robot3
id=103 and 104, search
id=105 and rotate

IP = "192.168.1.100"
Port = 13001
Get(IP, Port, 100, bytearray(2))
'''

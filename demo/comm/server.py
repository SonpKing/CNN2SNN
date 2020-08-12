# -*- coding: utf-8 -*-
"""
Created on Sat Jul  4 14:07:16 2020

@author: Editing
"""

import os
import socket
import sys
import traceback

from .connection import Connection

class ServerConnection(Connection):
    """
    服务器连接。
    
    Parameters
    ----------
    socket_ : socket.socket
        套接字。
    
    Notes
    -----
    当`ServerConnection`对象创建成功，一个符合给定参数的TCP连接已建立。
    """
    pass

class ServerListener:
    """
    服务器连接监听器。
    
    Parameters
    ----------
    server_name : str
        本地服务器域名或IP地址。
    server_port : int
        本地服务器监听端口号。
    timeout : float or None
        超时（以秒为单位）。
    num_clients : int
        远程客户端数量。
    
    Notes
    -----
    当`ServerListener`对象创建成功，服务器开始监听给定端口的连接请求。
    
    `timeout`指定底层套接字单次操作的超时，而非本类各个方法的超时。
    若`timeout`是正数，则本类的方法阻塞调用线程，并在底层套接字操作超时时抛出异常。
    若`timeout`是0或负数，则本类的方法不阻塞调用线程。
    若`timeout`是`None`，则本类的方法阻塞调用线程，直至操作完成为止。
    """
    
    class TimeoutException(Exception):
        def __init__(self):
            super(ServerListener.TimeoutException, self).__init__(
                'The accept operation timed out.' + os.linesep + \
                'Cause:' + os.linesep + \
                traceback.format_exc())
            _, self._cause, _ = sys.exc_info()
        
        def cause(self):
            """
            返回原始异常。
            
            Returns
            -------
            cause : socket.timeout
                原始异常。
            """
            return self._cause
    
    def accept(self):
        """
        接受连接请求。
        
        Returns
        -------
        connection : ServerConnection
            服务器连接。
        
        Raises
        ------
        ServerListener.TimeoutException
            接受连接操作超时。
        
        Notes
        -----
        `connection`的最长消息长度和超时符合ServerListener的参数。
        """
        try:
            socket_, _ = self._socket.accept()
        except socket.timeout:
            raise ServerListener.TimeoutException()
        socket_.settimeout(None)
        return ServerConnection(socket_)
    
    def close(self):
        """
        关闭连接。
        """
        self._socket.close()
    
    def __init__(self, server_name, server_port, timeout, num_clients):
        if timeout is not None and float(timeout) < 0.0:
            timeout = 0.0
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._socket.settimeout(timeout)
        self._socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._socket.bind((server_name, server_port))
        self._socket.listen(num_clients)
    
    def __del__(self):
        try:
            self.close()
        except:
            pass
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        return False

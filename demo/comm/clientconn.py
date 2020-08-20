# -*- coding: utf-8 -*-
"""
Created on Fri Jul  3 17:15:31 2020

@author: Editing
"""

import socket

from .connection import Connection

class ClientConnection(Connection):
    """
    客户端连接。
    
    Parameters
    ----------
    server_name : str
        远程服务器域名或IP地址。
    server_port : int
        远程服务器端口号。
    client_name : str, optional
        本地客户端域名或IP地址。
    client_port : int, optional
        本地客户端端口号。
    
    Notes
    -----
    当`ClientConnection`对象创建成功，一个符合给定参数的TCP连接已建立。
    """
    
    def __init__(self, server_name, server_port, **kwargs):
        args = {'address': (server_name, server_port), 'timeout': None}
        if 'client_name' not in kwargs:
            kwargs['client_name'] = ''
        if 'client_port' not in kwargs:
            kwargs['client_port'] = 0
        args['source_address'] = (kwargs['client_name'], kwargs['client_port'])
        super(ClientConnection, self).__init__(socket.create_connection(**args))

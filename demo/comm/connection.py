# -*- coding: utf-8 -*-
"""
Created on Sat Jul  4 08:37:51 2020

@author: Editing
"""

import socket
import struct

class Connection(object):
    """
    连接——用于收发数据。
    
    Parameters
    ----------
    socket_ : socket.socket
        套接字。
    """
    
    class DisconnectedException(Exception):
        def __init__(self):
            super(Connection.DisconnectedException, self).__init__(
                'The connection is closed by the peer.')
    
    def send(self, application_id, buffer, length = None):
        """
        发送消息。
        
        Parameters
        ----------
        application_id : int
            应用编号。
        buffer : bytearray
            消息载荷。
        length : int, optional
            消息载荷长度（以字节计）。
        
        Raises
        ------
        Connection.DisconnectedException
            连接已被对方关闭。
        
        Notes
        -----
        若`length`为不超过`buffer`长度的正整数，则消息载荷为`buffer`中的前`length`
        个字节；否则消息载荷为`buffer`中的所有字节。
        """
        application_id = int(application_id)
        if application_id < 0 or \
            application_id > Connection._APPLICATION_ID_MAXIMUM:
            raise ValueError(
                'application_id must be an integer within the range [0, {0}].'
                .format(Connection._APPLICATION_ID_MAXIMUM))
        if not isinstance(buffer, bytearray):
            raise TypeError('buffer must be a bytearray object.')
        if length is not None and int(length) > 0 and int(length) <= len(buffer):
            length_payload = int(length)
        else:
            length_payload = len(buffer)
        if length_payload <= 0 or \
            length_payload > Connection._LENGTH_PAYLOAD_MAXIMUM:
            raise ValueError(
                'The length of the mesage payload must be within the range ' \
                '[1, {0}].'
                .format(Connection._LENGTH_PAYLOAD_MAXIMUM))
        self._send_header(length_payload, application_id)
        self._do_send(buffer, length_payload)
    
    def receive(self):
        """
        接收消息。
        
        Returns
        -------
        application_id : int
            应用编号。
        data : bytearray
            消息载荷。
        
        Raises
        ------
        Connection.DisconnectedException
            连接已被对方关闭。
        
        Notes
        -----
        若没有消息可供接收，则`data`为`None`。
        """
        application_id = None
        data = None
        if self._receive_state == Connection._ReceiveState.HEADER:
            if self._do_receive():
                length_message, self._application_id = \
                    struct.unpack(
                        Connection._HEADER_FORMAT,
                        self._receive_buffer)
                # 首部的消息长度包含首部自身的长度。
                self._receive_state = Connection._ReceiveState.PAYLOAD
                self._receive_buffer = \
                    bytearray(length_message - Connection._LENGTH_HEADER)
                self._receive_offset = 0
        if self._receive_state == Connection._ReceiveState.PAYLOAD:
            if self._do_receive():
                application_id = self._application_id
                data = self._receive_buffer
                self._receive_state = Connection._ReceiveState.HEADER
                self._receive_buffer = self._receive_buffer_header
                self._receive_offset = 0
        elif self._receive_state != Connection._ReceiveState.HEADER:
            raise Exception('The receive state is invalid.')
        return application_id, data
    
    def close(self):
        """
        关闭连接。
        """
        self._socket.close()
    
    def __init__(self, socket_):
        if Connection._LENGTH_PAYLOAD_MAXIMUM <= 0:
            raise Exception(
                'The maximum length of message payload must be a positive ' \
                'integer.')
        if not isinstance(socket_, socket.socket):
            raise TypeError('socket_must be a socket.socket object。')
        self._socket = socket_
        self._send_buffer_header = bytearray(Connection._LENGTH_HEADER)
        self._receive_buffer_header = bytearray(Connection._LENGTH_HEADER)
        self._receive_state = Connection._ReceiveState.HEADER
        self._receive_buffer = self._receive_buffer_header
        self._receive_offset = 0
        self._application_id = None
    
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
    
    # 消息首部中消息长度的长度（以字节计）。
    _LENGTH_LENGTH = 4
    # 消息首部中应用编号的长度（以字节计）。
    _LENGTH_APPLICATION_ID = 4
    # 最大的应用编号。
    _APPLICATION_ID_MAXIMUM = 2 ** (_LENGTH_APPLICATION_ID << 3) - 1
    # 消息首部长度（以字节计）。
    _LENGTH_HEADER = _LENGTH_LENGTH + _LENGTH_APPLICATION_ID
    # 消息首部格式，用于消息首部的构造和解析。
    _HEADER_FORMAT = '<II'
    # 最长消息载荷的长度（以字节计）。
    _LENGTH_PAYLOAD_MAXIMUM = 2 ** (_LENGTH_LENGTH << 3) - 1 - _LENGTH_HEADER
    
    class _ReceiveState:
        HEADER = 1
        PAYLOAD = 2
    
    def _send_header(self, length_payload, application_id):
        # 首部的消息长度包含首部自身的长度。
        struct.pack_into(
            Connection._HEADER_FORMAT,
            self._send_buffer_header,
            0,
            Connection._LENGTH_HEADER + length_payload,
            application_id)
        self._do_send(
            self._send_buffer_header,
            Connection._LENGTH_HEADER)
    
    def _do_send(self, buffer, length):
        buffer = memoryview(buffer)
        offset = 0
        while offset < length:
            length_sent = self._socket.send(buffer[offset : length])
            if length_sent == 0:
                raise Connection.DisconnectedException()
            offset += length_sent
    
    def _do_receive(self):
        buffer = memoryview(self._receive_buffer)[self._receive_offset :]
        length_received = \
            self._socket.recv_into(
                buffer,
                len(self._receive_buffer) - self._receive_offset)
        if length_received == 0:
            raise Connection.DisconnectedException()
        self._receive_offset += length_received
        return self._receive_offset >= len(self._receive_buffer)

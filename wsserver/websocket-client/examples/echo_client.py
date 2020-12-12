from __future__ import print_function
#import websocket
import sys
# sys.path.append("..")
sys.path.append(".")
from websocket import *

if __name__ == "__main__":
    # websocket.enableTrace(True)
    # ws = websocket.create_connection("ws://echo.websocket.org/")
    enableTrace(True)
    ws = create_connection("ws://127.0.0.1:8800/")
    print("Sending 'Hello, World'...")
    ws.send("Hello, World")
    print("Sent")
    # print("Receiving...")
    # result = ws.recv()
    # print("Received '%s'" % result)
    ws.close()

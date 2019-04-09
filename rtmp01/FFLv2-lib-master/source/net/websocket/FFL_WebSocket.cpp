/*
*  This file is part of FFL project.
*
*  The MIT License (MIT)
*  Copyright (C) 2017-2018 zhufeifei All rights reserved.
*
*  FFL_WebSocke.cpp
*  Created by zhufeifei(34008081@qq.com) on 2019/02/24
*  https://github.com/zhenfei2016/FFLv2-lib.git
*
*  websocket封装
*/

#include "FFL_WebSocket.hpp"

namespace FFL {
#define  kMaxFrameSize  4096
	static void xorBytes(const uint8_t *in1, uint8_t *out, uint8_t maker[4], int size) {
		for (int i = 0; i < size; i++) {
			*out = in1[i] ^ maker[i % 4];
			out++;
		}
	}
	WebSocket::WebSocket(TcpClient* client, 
		NetStreamReader* mStream,
		bool isClient, 
		uint8_t* maskKey):
		mClient(client),		
		mStream(mStream),
		mIsClient(isClient){
		//
		//如果客户端 ,则发送的时候需要使用make进行xor
		//
		if (isClient &&maskKey) {
			mMarkerKey[0] = maskKey[0];
			mMarkerKey[1] = maskKey[1];
			mMarkerKey[2] = maskKey[2];
			mMarkerKey[3] = maskKey[3];
		} else {
			mMarkerKey[0] = 4;
			mMarkerKey[1] = 3;
			mMarkerKey[2] = 2;
			mMarkerKey[3] = 1;
		}
	}
	WebSocket::~WebSocket() {			
	}	
	//
	//  读frame头信息
	//
	bool WebSocket::recvFrameHead(WebsocketFrame& head) {
		if (!head.readHeader(mStream)) {
			return false;
		}

		if (mIsClient){
			//
			//  客户端接收到服务端的数据，应该不是加密的
			//
			return !head.mHaveMask;
		}

		//
		//  服务端接收客户端的数据式加密的
		//
		return head.mHaveMask;
	}
	//
	//  接收一帧数据
	//  buffer: 缓冲区 ， 
	//  bufferSize：这个输入缓冲区的大小。 当成功接收数据的情况下返回接收的数据大小
	//
	bool WebSocket::recvFrame(uint8_t* buffer, uint32_t* bufferSize) {		
		WebsocketFrame frame;	
		if (!recvFrameHead(frame)) {
			return false;
		}

		//
		//  如果数据过长>4096，则返回false
		uint8_t buf[4096] = {};
		size_t readed = 0;	
		if (!frame.readData(mStream, buffer, bufferSize)) {
			return false;
		}
		return true;
	}

	class WebSocketInputStream : public IOReader {
	public:
		WebSocketInputStream(NetStreamReader* stream,int64_t size):
			mStream(stream){			
				mSize=(uint32_t)size;
		}
		//
		//  读数据到缓冲区
		//  buf:缓冲区地址
		//  count:需要读的大小
		//  pReaded:实质上读了多少数据
		//  返回错误码  ： FFL_OK表示成功
		//
		 status_t read(uint8_t* buf, size_t count, size_t* pReaded) {
			if (mSize == mReaded) {
				return FFL_ERROR_EOF;
			}

			uint32_t requestSize =FFL_MIN((mSize - mReaded),count);
			if (requestSize == 0) {
				return FFL_FAILED;
			}			

			size_t readedSize = 0;
			int tryNum =0;
			while (++tryNum < 3) {
				uint32_t haveSize=mStream->getSize();
				if (haveSize >= requestSize) {
					mStream->readBytes((int8_t*)buf, requestSize);
					readedSize = requestSize;
					break;
				} else if (haveSize > 0) {
					mStream->readBytes((int8_t*)buf, haveSize);
					readedSize = haveSize;					
					break;
				}

				if (tryNum != 1) {
					FFL_sleep(tryNum * 5);
				}

				if (FFL_OK != mStream->pull(FFL_MIN((mSize - mReaded), 4096))) {
					continue;
				} 				
			}
			
			if (readedSize != 0) {
				mReaded += readedSize;
				if (pReaded) {
					*pReaded = readedSize;
				}				
				return (mSize == mReaded) ? FFL_ERROR_EOF : FFL_OK;
			}

			return FFL_FAILED;
		}


		NetStreamReader* mStream;
		uint32_t mReaded;
		uint32_t mSize;
	};

	IOReader* WebSocket::createInputStream() {		
		WebsocketFrame frame;
		if (!recvFrameHead(frame)) {
			return NULL;
		}

		return new WebSocketInputStream(mStream,frame.mPayloadLen);
	}
	void WebSocket::destroyInputStream(IOReader* reader) {
		WebSocketInputStream* stream = (WebSocketInputStream*)reader;
		FFL_SafeFree(stream);
	}

	//
	//  发送一帧数据
	//
	bool WebSocket::sendFrame(uint8_t opcode,const uint8_t* data, uint32_t len) {		
		WebsocketFrame frame;
		frame.FIN = true;
		frame.mOpcode = opcode;
		frame.mHaveMask = mIsClient?true:false;
		frame.mPayloadLen = len;
		frame.writeHeader(mClient);
		if (len <= 0) {
			return true;
		}

		static const int kBlockSize = 4096;
		if (frame.mHaveMask) {
			do {
				uint8_t outbuffer[kBlockSize] = {};
				uint32_t size = len > kBlockSize ? kBlockSize : len;
				xorBytes(data, outbuffer, mMarkerKey, size);
				mClient->write((void*)outbuffer, size, 0);
				len -= size;
			} while (len > kBlockSize);
				
		} else {
			mClient->write((void*)data, len, 0);
		}
		return true;
	}
	bool WebSocket::sendText(const char* text) {		
		return sendFrame(WebsocketFrame::OPCODE_TEXT, (uint8_t*)text,strlen(text));
	}
	bool WebSocket::sendBinary(const uint8_t* data, uint32_t len) {
		return sendFrame(WebsocketFrame::OPCODE_BINARY, data, len);
	}
	//
	//  读写二进制流
	//
	IOWriter* WebSocket::createOutputStream(uint32_t size) {
		FFL_ASSERT_LOG(0, "createOutputStream not implement");
		return NULL;
	}
	void WebSocket::destroyOutputStream(IOWriter* writer) {

	}

	bool WebSocket::sendPing() {
		return sendFrame(WebsocketFrame::OPCODE_PING, NULL,0);
	}
	bool WebSocket::sendPong() {
		return sendFrame(WebsocketFrame::OPCODE_PONG, NULL, 0);
	}
	bool WebSocket::sendBye() {
		return sendFrame(WebsocketFrame::OPCODE_BYE, NULL, 0);
	}	
}

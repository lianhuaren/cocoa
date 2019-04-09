/*
*  This file is part of FFL project.
*
*  The MIT License (MIT)
*  Copyright (C) 2017-2018 zhufeifei All rights reserved.
*
*  FFL_WebSocke.hpp
*  Created by zhufeifei(34008081@qq.com) on 2019/02/24
*  https://github.com/zhenfei2016/FFLv2-lib.git
*
*  websocket封装,用于读写websocket
*****************************************************************
* 只有当握手成功后才可以进行下面的读写操作的
******************************************************************
*/
#ifndef _FFL_WEBSOCKET_HPP_
#define _FFL_WEBSOCKET_HPP_

#include <net/FFL_TcpClient.hpp>
#include <net/FFL_NetStream.hpp>
#include "FFL_WebSocketFrame.hpp"

namespace FFL {
	class WebSocket{
	public:
		//
		//  client:已经建立好的websocket		
		//  mStream: client上读的数据，可能内部已经缓存了部分数据
		//  isClient:这是一个客户端还是服务端
		//  maskKey:xor的key值 
		//  
		WebSocket(TcpClient* client,			
			NetStreamReader* mStream,
			bool isClient,
			uint8_t* maskKey);
		~WebSocket();	
	public:
		//
		//  读frame头信息
		//
		bool recvFrameHead(WebsocketFrame& head);

		//  接收一帧数据
		//  buffer: 缓冲区 ， 
		//  bufferSize：这个输入缓冲区的大小。 当成功接收数据的情况下返回接收的数据大小
		//
		bool recvFrame(uint8_t* buffer, uint32_t* bufferSize);
		//
		//  读二进制流，可能会阻塞创建这个的，这个流式一帧websocket frame
		//
		IOReader* createInputStream();
		void destroyInputStream(IOReader* reader);
	public:
		//
		//  写
		//

		//
		//  发送一帧数据
		//
		bool sendFrame(uint8_t opcode, const uint8_t* data, uint32_t len);

		bool sendText(const char* text);
		bool sendBinary(const uint8_t* data, uint32_t len);
		//
		//  写二进制流,size :流大小
		//
		IOWriter* createOutputStream(uint32_t size);
		void destroyOutputStream(IOWriter* writer);

		bool sendPing();
		bool sendPong();
		bool sendBye();
	protected:
		TcpClient* mClient;
		NetStreamReader* mStream;	
		uint8_t mMarkerKey[4];
		bool mIsClient;
	};
}

#endif
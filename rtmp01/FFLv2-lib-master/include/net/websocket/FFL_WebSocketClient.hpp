/*
*  This file is part of FFL project.
*
*  The MIT License (MIT)
*  Copyright (C) 2017-2018 zhufeifei All rights reserved.
*
*  FFL_WebSocketClient.hpp
*  Created by zhufeifei(34008081@qq.com) on 2019/02/24
*  https://github.com/zhenfei2016/FFLv2-lib.git
*
*  用于websocket客户端连接 websocket服务器，然后进行发送接收数据桢
*/

#ifndef _FFL_WEBSOCKET_CLIENT_HPP_
#define _FFL_WEBSOCKET_CLIENT_HPP_

#include <FFL_RefBase.hpp>
#include <net/FFL_TcpClient.hpp>
#include <net/FFL_NetStream.hpp>
#include <FFL_ByteStream.hpp>

namespace FFL {		
	class WebSocket;
	class FFLIB_API_IMPORT_EXPORT WebSocketClient : public RefBase{
	public:
		WebSocketClient();
		virtual ~WebSocketClient();
	public:
		//
		//  连接host
		//
		bool connect(const char* host);
		bool connect(const char* ip,uint16_t port);
		bool isConnected() const;
		//
		//  获取建立的网络句柄
		//
		NetFD getFd();
		//
		//  关闭连接
		//
		void close();
	public:
		//
		//  发送握手请求，并等待握手成功应答
		//
		bool handshark(const char* path);	
		bool isHandshark() const;
	public:
		//*****************************************************************
		// 只有当握手成功后才可以进行下面的读写操作的
		//*****************************************************************

		//  接收一帧数据
		//  buffer: 缓冲区 ， 
		//  bufferSize：这个输入缓冲区的大小。 当成功接收数据的情况下返回接收的数据大小
		//
		bool recvFrame(uint8_t* buffer,uint32_t* bufferSize);
		//
		//  读二进制流，可能会阻塞创建这个的，这个流式一帧websocket frame
		//
		IOReader* createInputStream();
		void destroyInputStream(IOReader* reader);

		//
		//  发送一帧数据
		//
		bool sendFrame(uint8_t opcode,const uint8_t* data,uint32_t len);
		
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
		String mHost;
		WebSocket* mWebSocket;

		//
		//  是否已经连接成功了
		//
		volatile bool mIsConnected;
		// 
		//  是否已经握手成功了
		//
		volatile bool mIsHandShark;
	};
}
#endif

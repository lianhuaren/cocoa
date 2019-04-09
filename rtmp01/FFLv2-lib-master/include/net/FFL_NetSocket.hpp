/*
*  This file is part of FFL project.
*
*  The MIT License (MIT)
*  Copyright (C) 2017-2018 zhufeifei All rights reserved.
*
*  FFL_NetSocket.hpp   
*  Created by zhufeifei(34008081@qq.com) on 2018/07/14
*  https://github.com/zhenfei2016/FFL-v2.git
*  socket读写
*
*/
#ifndef _FFL_SOCKET_HPP_
#define _FFL_SOCKET_HPP_

#include <FFL_Io.hpp>

namespace FFL {
	class FFLIB_API_IMPORT_EXPORT CSocket : public IOReader,public IOWriter {
	public:
		enum Protocol{
			PROTOCOL_TCP = 0,
			PROTOCOL_UDP = 1,
		};
	public:
		//
		//  设置默认的tcp句柄
		//
		CSocket(NetFD fd=INVALID_NetFD);
		virtual ~CSocket();

		//
		//  设置socket句柄
		//
		void setFd(NetFD fd,CSocket::Protocol pro);
		NetFD getFd() const;		

		////////////////////////////////////////////////////////////////////////////////////////////////////////////////
		//
		//  udp协议使用的，设置写的目标地址
		//  获取最近读的来源地址
		//
		bool setWriteToAddr(const char* ip, uint16_t port);
		bool getReadFromAddr(char ip[32], uint16_t* port);
		//
		//  创建udp服务器
		//
		bool createUdpServer(const char* ip, uint16_t port);
		////////////////////////////////////////////////////////////////////////////////////////////////////////////////
		//
		//  tcp相关的，连接到服务器上，如果设置过fd则返回false
		//
		bool connect(const char* ip, uint16_t port);
		//
		//  关闭这个句柄
		//
		void close();
		//
		//  读数据到缓冲区
		//  buf:缓冲区地址
		//  count:需要读的大小
		//  pReaded:实质上读了多少数据
		//  返回错误码  ： FFL_OK表示成功
		//
		virtual status_t read(uint8_t* buf, size_t count, size_t* pReaded);
		//
		//  写数据到文件中
		//  buf:缓冲区地址
		//  count:缓冲区大小
		//  pWrite:实质上写了多少数据
		//  返回错误码  ： FFL_OK表示成功
		//
		virtual status_t write(const void* buf, size_t count, size_t* pWrite) ;
		//
		//  写数据到文件中
		//  bufVec:缓冲区地址,数组
		//  count:数组大小
		//  pWrite:实质上写了多少数据
		//  返回错误码  ： FFL_OK表示成功
		//
		virtual status_t writeVec(const BufferVec* bufVec, int count, size_t* pWrite) ;
	protected:
		DISABLE_COPY_CONSTRUCTORS(CSocket);
	protected:
		//
		//  是否tcp协议的
		//
		Protocol mProto;
		NetFD mFd;

		struct UdpParam {
			char mWriteToIP[32];
			uint16_t mWriteToPort;

			char mReadFromIP[32];
			uint16_t mReadFromPort;
		};
		UdpParam* mUdpParmas;
	};
}
#endif

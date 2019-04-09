/*
*  This file is part of FFL project.
*
*  The MIT License (MIT)
*  Copyright (C) 2017-2018 zhufeifei All rights reserved.
*
*  FFL_TcpClient.hpp
*  Created by zhufeifei(34008081@qq.com) on 2018/07/14
*  https://github.com/zhenfei2016/FFLv2-lib.git
*  网络客户端定义
*
*/
#ifndef _FFL_TCP_CLIENT_HPP_
#define _FFL_TCP_CLIENT_HPP_

#include <net/FFL_NetSocket.hpp>
#include <FFL_RefBase.hpp>

namespace FFL {
	class FFLIB_API_IMPORT_EXPORT TcpClient : public IOReader, public IOWriter {
    public:
        //
        //  tcp相关的，连接到服务器上，如果设置过fd则返回false
        //
        static status_t connect(const char* ip, uint16_t port,TcpClient& client);

	public:
		TcpClient(NetFD fd=INVALID_NetFD);
		virtual ~TcpClient();

		//
		// 保存一些用户数据
		//
		void setUserdata(void* priv);
		void* getUserdata();

		NetFD getFd();

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
		virtual status_t write(const void* buf, size_t count, size_t* pWrite);
		//
		//  写数据到文件中
		//  bufVec:缓冲区地址,数组
		//  count:数组大小
		//  pWrite:实质上写了多少数据
		//  返回错误码  ： FFL_OK表示成功
		//
		virtual status_t writeVec(const BufferVec* bufVec, int count, size_t* pWrite);		
	public:
		void close();

		
	protected:
		friend class HttpClient;
		friend class WebSocketClient;
		friend class NetStreamReader;
		CSocket mSocket;
		void* mPriv;
	};	

	
}

#endif
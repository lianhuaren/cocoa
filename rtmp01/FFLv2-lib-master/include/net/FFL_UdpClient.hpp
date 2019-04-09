/*
*  This file is part of FFL project.
*
*  The MIT License (MIT)
*  Copyright (C) 2017-2018 zhufeifei All rights reserved.
*
*  FFL_UdpClient.hpp
*  Created by zhufeifei(34008081@qq.com) on 2018/07/14
*  https://github.com/zhenfei2016/FFLv2-lib.git
*  udp基础类
*
*/

#include <net/FFL_NetSocket.hpp>

namespace FFL {
	class FFLIB_API_IMPORT_EXPORT UdpClient : public IOReader, public IOWriter {
	public:
		UdpClient(NetFD fd);
		UdpClient(const char* remoteIp, uint16_t remotePort);
		virtual ~UdpClient();

		//
		//  udp协议使用的，设置写的目标地址
		//  获取最近读的来源地址
		//
		bool setWriteToAddr(const char* ip, uint16_t port);
		bool getReadFromAddr(char ip[32], uint16_t* port);
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

		void close();
	protected:
		CSocket mSocket;
	};
}

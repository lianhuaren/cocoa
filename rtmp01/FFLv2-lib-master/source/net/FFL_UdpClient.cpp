/*
*  This file is part of FFL project.
*
*  The MIT License (MIT)
*  Copyright (C) 2017-2018 zhufeifei All rights reserved.
*
*  FFL_UdpClient.cpp
*  Created by zhufeifei(34008081@qq.com) on 2018/07/14
*  https://github.com/zhenfei2016/FFLv2-lib.git
*  udp基础类
*
*/

#include <net/FFL_Net.h>
#include <net/FFL_UdpClient.hpp>

namespace FFL {
   

    UdpClient::UdpClient(NetFD fd){
		mSocket.setFd(fd, CSocket::PROTOCOL_UDP);
    }
	UdpClient::UdpClient(const char* remoteIp,uint16_t remotePort) {
		NetFD fd = FFL_socketCreateUdp();		
		FFL_ASSERT_LOG(fd != INVALID_NetFD,"UdpClient()");
		
		mSocket.setFd(fd, CSocket::PROTOCOL_UDP);
		mSocket.setWriteToAddr(remoteIp, remotePort);
	}
    UdpClient::~UdpClient(){		
    }

	//
	//  udp协议使用的，设置写的目标地址
	//  获取最近读的来源地址
	//
	bool UdpClient::setWriteToAddr(const char* ip, uint16_t port) {
		return mSocket.setWriteToAddr(ip,port);
	}
	bool UdpClient::getReadFromAddr(char ip[32], uint16_t* port) {
		return mSocket.getReadFromAddr(ip,port);
	}

	//
	//  读数据到缓冲区
	//  buf:缓冲区地址
	//  count:需要读的大小
	//  pReaded:实质上读了多少数据
	//  返回错误码  ： FFL_OK表示成功
	//
	status_t UdpClient::read(uint8_t* buf, size_t count, size_t* pReaded) {
		return  mSocket.read(buf, count, pReaded);
	}
	//
	//  写数据到文件中
	//  buf:缓冲区地址
	//  count:缓冲区大小
	//  pWrite:实质上写了多少数据
	//  返回错误码  ： FFL_OK表示成功
	//
	status_t UdpClient::write(const void* buf, size_t count, size_t* pWrite) {
		return  mSocket.write(buf, count, pWrite);
	}
	//
	//  写数据到文件中
	//  bufVec:缓冲区地址,数组
	//  count:数组大小
	//  pWrite:实质上写了多少数据
	//  返回错误码  ： FFL_OK表示成功
	//
	status_t UdpClient::writeVec(const BufferVec* bufVec, int count, size_t* pWrite) {
		return  mSocket.writeVec(bufVec, count, pWrite);
	}
	void UdpClient::close() {
		mSocket.close();
	}



	
}

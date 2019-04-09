/*
*  This file is part of FFL project.
*
*  The MIT License (MIT)
*  Copyright (C) 2017-2018 zhufeifei All rights reserved.
*
*  FFL_NetSocket.cpp
*  Created by zhufeifei(34008081@qq.com) on 2018/07/14
*  https://github.com/zhenfei2016/FFLv2-lib.git
*  socket读写
*
*/

#include <net/FFL_Net.h>
#include <net/FFL_NetSocket.hpp>


namespace FFL {
	CSocket::CSocket(NetFD fd):mFd(fd){
		FFL_socketInit();
		mProto = PROTOCOL_TCP;
		mUdpParmas = NULL;
	}
	CSocket::~CSocket() {
		FFL_SafeFree(mUdpParmas);
	}
	//
	//  设置socket句柄
	//
	void CSocket::setFd(NetFD fd, CSocket::Protocol pro) {
		mFd = fd;
		mProto = pro;

		FFL_SafeFree(mUdpParmas);
		mUdpParmas = new UdpParam();
		memset((void*)mUdpParmas, 0, sizeof(mUdpParmas));

	}
	NetFD CSocket::getFd() const {
		return mFd;
	}

	//
	//  udp协议使用的，设置写的目标地址
	//  获取最近读的来源地址
	//
	bool CSocket::setWriteToAddr(const char* ip, uint16_t port) {
		if (mProto != PROTOCOL_UDP) {
			return false;
		}

		if (ip == NULL || port == 0) {
			return false;
		}

		memcpy(mUdpParmas->mWriteToIP, ip, FFL_MAX((strlen(ip )+ 1), 31));
		mUdpParmas->mWriteToPort = port;

		return true;
	}
	bool CSocket::getReadFromAddr(char ip[32], uint16_t* port) {
		if (mProto != PROTOCOL_UDP) {
			return false;
		}

		memcpy(ip, mUdpParmas->mReadFromIP, 32);
		if (port) {
			*port = mUdpParmas->mReadFromPort;
		}
		return true;
	}
	bool CSocket::createUdpServer(const char* ip, uint16_t port) {
		if (mFd != 0) {
			return false;
		}

		NetFD fd = 0;
		if (FFL_OK != FFL_socketAnyAddrUdpServer(port, &fd)) {
			return false;
		}
		setFd(fd, FFL::CSocket::PROTOCOL_UDP);
		return true;
	}
	//
	//  tcp相关的，连接到服务器上，如果设置过fd则返回false
	//
	bool CSocket::connect(const char* ip, uint16_t port) {
		if (mFd != 0) {
			return false;
		}

		NetFD fd = FFL_socketCreateTcp();;
		if (fd == 0) {
			return false;
		}

		FFL_socketSetSendTimeout(fd, 15 * 1000);
		FFL_socketSetRecvTimeout(fd, 15 * 1000);
		if (FFL_OK == FFL_socketNetworkTcpClient(ip, port, &fd)) {
			mFd = fd;
			mProto = PROTOCOL_TCP;
			return true;
		}

		return false;
	}
	//
	//  关闭这个句柄
	//
	void CSocket::close() {
		if (mFd != 0) {
			FFL_socketClose(mFd);
			mFd = 0;
		}
	}
	//
	//  读数据到缓冲区
	//  buf:缓冲区地址
	//  count:需要读的大小
	//  pReaded:实质上读了多少数据
	//  返回错误码  ： FFL_OK表示成功
	//
	status_t CSocket::read(uint8_t* buf, size_t count, size_t* pReaded) {
		if (mProto == PROTOCOL_TCP) {
			return FFL_socketRead(mFd, buf, count, pReaded);
		}

		return FFL_socketReadFrom(mFd, buf, count, pReaded,
			mUdpParmas->mReadFromIP,
			&mUdpParmas->mReadFromPort);
	}
	//
	//  写数据到文件中
	//  buf:缓冲区地址
	//  count:缓冲区大小
	//  pWrite:实质上写了多少数据
	//  返回错误码  ： FFL_OK表示成功
	//
	status_t CSocket::write(const void* buf, size_t count, size_t* pWrite) {
		if (mProto == PROTOCOL_TCP) {
			return FFL_socketWrite(mFd, buf, count, pWrite);
		}

		return FFL_socketWriteTo(mFd, buf, count, pWrite,
			mUdpParmas->mWriteToIP,
			mUdpParmas->mWriteToPort);
	}
	//
	//  写数据到文件中
	//  bufVec:缓冲区地址,数组
	//  count:数组大小
	//  pWrite:实质上写了多少数据
	//  返回错误码  ： FFL_OK表示成功
	//
	status_t CSocket::writeVec(const BufferVec* bufVec, int count, size_t* pWrite) {
		status_t ret = FFL_OK;
		size_t nWriteAll = 0;
		size_t nWrite;
		for (int32_t i = 0; i < count; i++) {
			const BufferVec* pBuf = bufVec + i;

			if (mProto == PROTOCOL_TCP) {
				ret = FFL_socketWrite(mFd, pBuf->data, pBuf->size, &nWrite);
			} else {
				ret = FFL_socketWriteTo(mFd, pBuf->data, pBuf->size, &nWrite,
					mUdpParmas->mWriteToIP,
					mUdpParmas->mWriteToPort);
			}

			if (FFL_OK != ret) {
				break;
			}
			nWriteAll += nWrite;
		}
		
		if (pWrite) {
			*pWrite = nWriteAll;
		}
		return ret;
	}
}
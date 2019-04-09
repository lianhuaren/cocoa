/*
*  This file is part of FFL project.
*
*  The MIT License (MIT)
*  Copyright (C) 2017-2018 zhufeifei All rights reserved.
*
*  FFL_SocketPair.cpp
*  Created by zhufeifei(34008081@qq.com) on 2018/11/24
*  https://github.com/zhenfei2016/FFLv2-lib.git
*  socketpair模拟定义
*
*/

#include "FFL_SocketPair.hpp"

#include <net/FFL_NetSocket.hpp>
#include "base/internalSocket.h"
#if ANDROID
#include <arpa/inet.h>
#endif


namespace FFL {
	SocketPair::SocketPair() {
		mFd[0] = INVALID_NetFD;
		mFd[1] = INVALID_NetFD;
	}
	SocketPair::~SocketPair() {
	}

#if WIN32
	//
	//
	//
	static bool createLoopbackPair(NetFD fd[2]) {		
		int size;
		struct sockaddr_in addr;
		NetFD serverFd = INVALID_NetFD;
		uint16_t serverPort = 0;
		fd[0] = fd[1] = INVALID_NetFD;
		
		//
		//
		//
		serverFd=FFL_socketCreateTcp();
		memset(&addr, 0, sizeof(struct sockaddr_in));
		addr.sin_family = AF_INET;
		addr.sin_addr.s_addr = inet_addr("127.0.0.1");
		if (bind(serverFd, (struct sockaddr *)&addr, sizeof(struct sockaddr_in)) != 0) {
			goto fail;
		}
		memset(&addr, 0, sizeof(struct sockaddr_in));
		size = sizeof(struct sockaddr_in);
		if (getsockname(serverFd, (struct sockaddr*)&addr, &size) != 0) {
			goto fail;
		}	
		serverPort = ntohs(addr.sin_port);
		if (listen(serverFd, 1) != 0) {
			goto fail;
		}

		//
		//
		//
		fd[0] = socket(PF_INET, SOCK_STREAM, 0);
		if (fd[0] == INVALID_NetFD){
			goto fail;
		}
		memset(&addr, 0, sizeof(struct sockaddr_in));
		addr.sin_family = AF_INET;
		if (bind(fd[0], (struct sockaddr *)&addr, sizeof(struct sockaddr_in)) != 0){
			goto fail;;
		}
		memset(&addr, 0, sizeof(struct sockaddr_in));
		addr.sin_family = AF_INET;
		addr.sin_addr.s_addr = inet_addr("127.0.0.1");
		addr.sin_port = htons(serverPort);
		if (connect(fd[0], (struct sockaddr *)&addr, sizeof(struct sockaddr_in)) != 0){
			goto fail;
		}

		//
		//
		//
		memset(&addr, 0, sizeof(struct sockaddr_in));
		size = sizeof(struct sockaddr_in);
		fd[1] = accept(serverFd, (struct sockaddr *)&addr, &size);
		if (fd[1] == INVALID_NetFD){
			goto fail;
		}		

		FFL_socketClose(serverFd);
		return true;

	fail:
		if (fd[0] != INVALID_NetFD) {
			FFL_socketClose(fd[0]);
			fd[0] = INVALID_NetFD;
		}

		if (fd[1] != INVALID_NetFD) {
			FFL_socketClose(fd[1]);
			fd[1] = INVALID_NetFD;
		}
		FFL_socketClose(serverFd);
		return false;
	}
#endif

	bool SocketPair::create() {
		if (mFd[0] == INVALID_NetFD && mFd[1] == INVALID_NetFD) {
#if WIN32
            return createLoopbackPair(mFd);
#else
            int sockets[2];
            if(0!=socketpair(AF_LOCAL, SOCK_STREAM, 0,sockets)){
                //int err=SOCKET_ERRNO();
                return false;
            }
            if(sockets[0]!=INVALID_NetFD ){
                mFd[0]=sockets[0];
                mFd[1]=sockets[1];
            }
            return  true;
#endif
		}
		return false;
	}
	void SocketPair::destroy() {
		if (mFd[0] != INVALID_NetFD) {
			FFL_socketClose(mFd[0]);
			mFd[0] = INVALID_NetFD;
		}

		if (mFd[1] != INVALID_NetFD) {
			FFL_socketClose(mFd[1]);
			mFd[1] = INVALID_NetFD;
		}
	}
	NetFD SocketPair::getFd0() const {
		return mFd[0];
	}
	NetFD SocketPair::getFd1() const {
		return mFd[1];
	}
	//
	//
	//
	bool SocketPair::writeFd0(const uint8_t* data, size_t size, size_t* writedSize) {
		if (mFd[0] == INVALID_NetFD || data ==NULL || size ==0 ) {
			return false;
		}

		CSocket socket(mFd[0]);
		return socket.write((void*)data, size, writedSize) == FFL_OK;
	}
	//
	//
	//
	bool SocketPair::readFd1(uint8_t* data, size_t size, size_t* readedSize) {
		if (mFd[0] == INVALID_NetFD || data==NULL || size ==0 ) {
			return false;
		}

		CSocket socket(mFd[0]);
		return socket.read((uint8_t*)data, size, readedSize) == FFL_OK;
	}
}

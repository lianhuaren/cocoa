 /*
 *  This file is part of FFL project.
 *
 *  The MIT License (MIT)
 *  Copyright (C) 2017-2018 zhufeifei All rights reserved.
 *
 *  FFL_Net.c
 *  Created by zhufeifei(34008081@qq.com) on 2017/8/12.
 *  https://github.com/zhenfei2016/FFLv2-lib.git
 *  网络socket公用函数
 *
 */

#include "internalSocket.h"
#include <net/FFL_Net.h>
#include "internalLogConfig.h"


static int gSocketInited=0;

void initializeNetModule() {
	FFL_socketInit();
}
void terminateNetModule() {
	FFL_socketUninit();
}
void FFL_socketInit(){
	if (!gSocketInited) {
		SOCKET_SETUP();
		gSocketInited++;
	}
}
void FFL_socketUninit(){
	if (gSocketInited) {
		SOCKET_CLEANUP();
		gSocketInited=0;
	}
}
/*
*  创建socket
*/
NetFD FFL_socketCreate(int type) {
	NetFD fd = socket(AF_INET, type, 0);

	if (fd > 0) {
#ifdef ANDROID
		int flags = fcntl(fd, F_GETFD);
		flags |= FD_CLOEXEC;
		fcntl(fd, F_SETFD, flags);
#endif
	}
	return fd;
}
/*
*  关闭socket
*/
void FFL_socketClose(NetFD fd) {
	FFL_SOCKET_CLOSE(fd);
}
/*
*  accept一个客户端上来
*/
SOCKET_STATUS FFL_socketAccept(NetFD serverfd, NetFD* clientFd){
	int err = FFL_SOCKET_OK;
	int socketErr = 0;
	struct sockaddr_in addr;
	int addrlen = sizeof(struct sockaddr_in);
	int sockfd = accept(serverfd, (struct sockaddr *) &addr, (socklen_t*)&addrlen);
	if (sockfd < 0) {
		socketErr = SOCKET_ERRNO();
		INTERNAL_FFL_LOG_WARNING("FFL_socketAccept error=%d", socketErr);
		//是否超时超时
		if (socketErr == SOCKET_AGAIN ||
			socketErr == SOCKET_ETIME ||
			socketErr == SOCKET_ECONNRESET) {
			err = FFL_ERROR_SOCKET_TIMEOUT;
		}else {
			err = FFL_ERROR_SOCKET_FAILED;
		}
	}

	if (clientFd) {
		*clientFd = sockfd;
	}
	return err;
}
/*
* 读 ,
* readed : 获取读取了数据量
* 失败返回，FFL_ERROR_SOCKET_XX
* 成功返回  FFL_ERROR_SUCCESS
* */
SOCKET_STATUS FFL_socketRead(NetFD fd, void* buffer, size_t size,size_t* readed){
	int socketError=0;
	int nbRead = recv(fd, buffer, size, 0);
	if (nbRead > 0) {
		if (readed) {
			*readed = nbRead;
		}
		return  FFL_SOCKET_OK;
	}

	if (nbRead < 0  )	{
		socketError =SOCKET_ERRNO();	
		INTERNAL_FFL_LOG_WARNING("FFL_socketRead error=%d", socketError);
		//读写超时
		if (socketError == SOCKET_AGAIN ||
			socketError == SOCKET_ETIME ||
			socketError == SOCKET_ECONNRESET) {
			return FFL_ERROR_SOCKET_TIMEOUT;
		} else {
			return FFL_ERROR_SOCKET_READ_EX + socketError;
		}
	}

	/* 
	 *  服务端关闭了 
	 */
	if (nbRead == 0) {
		INTERNAL_FFL_LOG_WARNING("FFL_socketRead nb_read=0 error=%d",SOCKET_ECONNRESET);
		return FFL_ERROR_SOCKET_READ_EX + SOCKET_ECONNRESET;
	}

	return FFL_ERROR_SOCKET_READ;
}
SOCKET_STATUS FFL_socketReadFrom(NetFD fd, void* buffer, size_t size, size_t* readed, char* srcIp, uint16_t* srcPort) {
	int socketError = 0;
	int nbRead =0;
	int fromlen = sizeof(struct sockaddr_in);
	struct sockaddr_in srcAddr;
	FFL_Zerop(&srcAddr);
	nbRead = recvfrom(fd,(char*)buffer, size, 0,
                          (struct sockaddr *)(&srcAddr),
                          (socklen_t*) &fromlen);
	if (nbRead > 0) {
		if (readed) {
			*readed = nbRead;
		}

		if (srcIp) {
			const char* peerIp = inet_ntoa(srcAddr.sin_addr);
			int maxLen = FFL_MIN((strlen(peerIp)+1),16);
			memcpy(srcIp, peerIp, maxLen);
		}

		if (srcPort) {
			*srcPort = ntohs(srcAddr.sin_port);
		}
		return  FFL_SOCKET_OK;
	}

	if (nbRead < 0) {
		socketError = SOCKET_ERRNO();
		INTERNAL_FFL_LOG_WARNING("FFL_socketReadFrom error=%d", socketError);
		//读写超时
		if (socketError == SOCKET_AGAIN ||
			socketError == SOCKET_ETIME ||
			socketError == SOCKET_ECONNRESET) {
			return FFL_ERROR_SOCKET_TIMEOUT;
		}
		else {
			return FFL_ERROR_SOCKET_READ_EX + socketError;
		}
	}

	/*
	*  服务端关闭了
	*/
	if (nbRead == 0) {
		INTERNAL_FFL_LOG_WARNING("FFL_socketRead nb_read=0 error=%d", SOCKET_ECONNRESET);
		return FFL_ERROR_SOCKET_READ_EX + SOCKET_ECONNRESET;
	}

	return FFL_ERROR_SOCKET_READ;
}
/*
 * 写 ,
 * writed : 写成功了多少数据
 * 失败返回，FFL_ERROR_SOCKET_XX
 * 成功返回  FFL_ERROR_SUCCESS
 * */
SOCKET_STATUS FFL_socketWrite(NetFD fd, const  void* buffer, size_t size,size_t* writed){
	int socketError=0;
	int nbWrite = send(fd, (char*)buffer, size, 0);
	if (nbWrite > 0) {
		if (writed)
			*writed = nbWrite;
		return  FFL_ERROR_SUCCESS;
	}		

	if (nbWrite < 0  )	{
		socketError=SOCKET_ERRNO();
		INTERNAL_FFL_LOG_WARNING("FFL_socketWrite error=%d", socketError);
		/*
		 * 读写超时
	 	 */
		if (socketError == SOCKET_AGAIN ||
			socketError == SOCKET_ETIME ||
			socketError == SOCKET_ECONNRESET) {
			return FFL_ERROR_SOCKET_TIMEOUT;
		}
		else {
			return FFL_ERROR_SOCKET_WRITE_EX + socketError;
		}
	}

	return FFL_ERROR_SOCKET_WRITE;
}

SOCKET_STATUS FFL_socketWriteTo(NetFD fd, const  void* buffer, size_t size, size_t* writed, const char* destIp, uint16_t destPort) {
	int socketError = 0;
	int nbWrite=0;
	struct sockaddr_in destAddr;

	if (destIp == NULL || strlen(destIp) > 32) { 
		return FFL_FAILED;
	}

	destAddr.sin_family = AF_INET;      
	destAddr.sin_port = htons(destPort);	
	destAddr.sin_addr.s_addr = inet_addr(destIp);
//EOPNOTSUPP
	nbWrite = sendto(fd,(char*) buffer, size, 0,(struct sockaddr*)&destAddr,sizeof(struct sockaddr_in));	
	if (nbWrite > 0) {
		if (writed)
			*writed = nbWrite;
		return  FFL_ERROR_SUCCESS;
	}	

	if (nbWrite < 0) {
		socketError = SOCKET_ERRNO();
        FFL_LOG_WARNING("FFL_socketWriteTo error=%d  ip=%s:%d", socketError,destIp,destPort);
		/*
		* 读写超时
		*/
		if (socketError == SOCKET_AGAIN ||
			socketError == SOCKET_ETIME ||
			socketError == SOCKET_ECONNRESET) {
			return FFL_ERROR_SOCKET_TIMEOUT;
		}
		else {
			return FFL_ERROR_SOCKET_WRITE_EX + socketError;
		}
	}

	return FFL_ERROR_SOCKET_WRITE;
}
/*
*  设置接收超时值
*/
SOCKET_STATUS FFL_socketSetRecvTimeout(NetFD fd, int64_t ms){
	struct timeval tv = { (int32_t)(ms / 1000) , (int32_t)((ms % 1000)*1000) };
	if (setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO,(const char*)( &tv), sizeof(tv)) == -1)
	{
		return FFL_ERROR_SOCKET_FAILED;
	}
	return FFL_SOCKET_OK;
}

/*
*  设置发送超时值
*/
SOCKET_STATUS FFL_socketSetSendTimeout(NetFD fd, int64_t ms){
	struct timeval tv = { (int32_t)(ms / 1000) , (int32_t)((ms % 1000)*1000) };
	if (setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, (const char*)(&tv), sizeof(tv)) == -1)
	{
		return FFL_ERROR_SOCKET_FAILED;
	}
	return FFL_SOCKET_OK;
}
/*
*  设置发送不延迟发送
*/
SOCKET_STATUS FFL_socketSetNodelay(NetFD fd, int yes)
{
#if defined(WIN32)
	int flag = (yes != 0) ? 1 : 0;
	int nSendBuf = 0;
	setsockopt(fd, SOL_SOCKET, SO_SNDBUF, (const char*)&nSendBuf, sizeof(int));
	setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, (char *)&flag, sizeof(flag));
#else
	int enable = 1;
	setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, (void*)&enable, sizeof(enable));
#endif
	return FFL_SOCKET_OK;
}
/*
* 分解host和port
* www.123.com:4000
* host=www.123.com
* port=4000
*/
SOCKET_STATUS FFL_socketPaserHost(const char* url, char* host, int32_t* port){
	const char* src = url;
	while (*src) {
		if (*src == ':') {
			*port = atoi(src + 1);			
			break;
		}
		*host++ = *src++;
	}
	*host = 0;
	return FFL_OK;
}

/*
*  获取本地的地址，返回获取的数量
*  hostlist : 如果多个地址则使用;分割开
*/
int32_t FFL_socketLocalAddr(char* hostlist, int32_t bufSize) {
	char hostname[128];
	struct hostent *hent;
	struct in_addr* addr = 0;

	gethostname(hostname, sizeof(hostname));	
	hent = gethostbyname(hostname);
	if (0 == hent){
	    return 0;
	}	
	
	addr = ((struct in_addr*)hent->h_addr);
	strncpy(hostlist, inet_ntoa(*addr), bufSize - 1);
	return 1;
}

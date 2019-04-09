/*
*  This file is part of FFL project.
*
*  The MIT License (MIT)
*  Copyright (C) 2017-2018 zhufeifei All rights reserved.
*
*  FFL_Net.h
*  Created by zhufeifei(34008081@qq.com) on 2017/8/12.
*  https://github.com/zhenfei2016/FFL-v2.git
*  网络socket公用函数
*
*/

#ifndef _FFL_SOCKET_H_
#define _FFL_SOCKET_H_

#include <FFL_Core.h>
#include <net/FFL_NetConst.h>

#ifdef WIN32
#include <winsock2.h>
typedef int  socklen_t;
#else
#include <sys/socket.h>
#endif

#ifdef __cplusplus 
extern "C" {
#endif
	/* socket库初始化 */
	FFLIB_API_IMPORT_EXPORT void FFL_socketInit();
	FFLIB_API_IMPORT_EXPORT void FFL_socketUninit();
	
	/*
	*  创建socket
	*/
	FFLIB_API_IMPORT_EXPORT NetFD FFL_socketCreate(int type);
    #define FFL_socketCreateTcp() FFL_socketCreate(SOCK_STREAM)
    #define FFL_socketCreateUdp() FFL_socketCreate(SOCK_DGRAM)
	/*
	*关闭句柄
	*/
	FFLIB_API_IMPORT_EXPORT void FFL_socketClose(NetFD fd);

    /*
	 *  连接到loopback的服务器，返回句柄 
	 *  如果fd指向的为NULL 则内部会进行socket的创建
	 */
	FFLIB_API_IMPORT_EXPORT SOCKET_STATUS FFL_socketLoopbackClient(int port, int type, NetFD*fd);
    #define FFL_socketLoopbackTcpClient(port,fd) FFL_socketLoopbackClient(port,SOCK_STREAM,fd)
    #define FFL_socketLoopbackUdpClient(port,fd) FFL_socketLoopbackClient(port,SOCK_DGRAM,fd)
    /*  
	 *  本地loopback的服务器，监听连接
	 *  如果fd指向的为NULL 则内部会进行socket的创建
	 */
	FFLIB_API_IMPORT_EXPORT SOCKET_STATUS FFL_socketLoopbackServer(int port, int type,NetFD* fd);
    #define FFL_socketLoopbackTcpServer(port,fd) FFL_socketLoopbackServer(port,SOCK_STREAM,fd);  
    #define FFL_socketLoopbackUdpServer(port,fd) FFL_socketLoopbackServer(port,SOCK_DGRAM,fd);·
	/*
	 *  连接到服务器，返回句柄fd
	 *  如果fd指向的为NULL 则内部会进行socket的创建
	 */
	FFLIB_API_IMPORT_EXPORT SOCKET_STATUS FFL_socketNetworkClient(const char *host, int port, int type, NetFD* fd);
	#define FFL_socketNetworkTcpClient(host,port,fd) FFL_socketNetworkClient(host,port,SOCK_STREAM,fd)
	#define FFL_socketNetworkUdpClient(host,port,fd) FFL_socketNetworkClient(host,port,SOCK_DGRAM,fd)

	/* 
	 *  本地的服务器，监听连接 
	 *  如果fd指向的为NULL 则内部会进行socket的创建
	 */
	FFLIB_API_IMPORT_EXPORT SOCKET_STATUS FFL_socketAnyAddrServer(int port, int type,NetFD* fd);
	#define FFL_socketAnyAddrTcpServer(port,fd) FFL_socketAnyAddrServer(port,SOCK_STREAM,fd)
	#define FFL_socketAnyAddrUdpServer(port,fd) FFL_socketAnyAddrServer(port,SOCK_DGRAM,fd)
    /*  
	 * 服务端accept一个客户端，当一个客户端连接上来的时候，这个会返回的  clientfd
	 */
	FFLIB_API_IMPORT_EXPORT SOCKET_STATUS FFL_socketAccept(NetFD serverfd, NetFD* clientfd);
    /*
     * 网络读 
     * readed : 获取读取了数据量
     * 失败返回，FFL_ERROR_SOCKET_XX
     * 成功返回  FFL_ERROR_SUCCESS
     * */
	FFLIB_API_IMPORT_EXPORT SOCKET_STATUS FFL_socketRead(NetFD fd, void* buffer, size_t size,size_t* readed);
	/*
	*  udp上读
	*  srcIp :源地址，需要保证长度啊 
	*  srcPort :源端口
	*/
	FFLIB_API_IMPORT_EXPORT SOCKET_STATUS FFL_socketReadFrom(NetFD fd, void* buffer, size_t size, size_t* readed, char* srcIp,uint16_t* srcPort);

	/*
	 * 网络写 
	 * writed : 写成功了多少数据
	 * 失败返回，FFL_ERROR_SOCKET_XX
	 * 成功返回  FFL_ERROR_SUCCESS
	 * */
	FFLIB_API_IMPORT_EXPORT SOCKET_STATUS FFL_socketWrite(NetFD fd, const void* buffer, size_t size,size_t* writed);
	FFLIB_API_IMPORT_EXPORT SOCKET_STATUS FFL_socketWriteTo(NetFD fd, const  void* buffer, size_t size, size_t* writed, const char* destIp, uint16_t destPort);

    /*
	 * 设置发送，接收超时时间
	 */
	FFLIB_API_IMPORT_EXPORT SOCKET_STATUS FFL_socketSetRecvTimeout(NetFD fd, int64_t ms);
	FFLIB_API_IMPORT_EXPORT SOCKET_STATUS FFL_socketSetSendTimeout(NetFD fd, int64_t ms);
	/* 
	 *  设置不延迟发送 (禁用粘包)
	 */
	FFLIB_API_IMPORT_EXPORT SOCKET_STATUS FFL_socketSetNodelay(NetFD fd, int yes);
	/*
	* 分解host和port
	* www.123.com:4000   
	* host=www.123.com
	* port=4000
	*/
	FFLIB_API_IMPORT_EXPORT SOCKET_STATUS FFL_socketPaserHost(const char* hostport, char* host, int32_t* port);
	/*
	*  获取本地的地址，返回获取的数量
	*  hostlist : 如果多个地址则使用;分割开
	*/
	FFLIB_API_IMPORT_EXPORT int32_t FFL_socketLocalAddr(char* hostlist, int32_t bufSize);
	/*
	*  select模式测试fdlist这一组socket是否有可以读的
	*  fdNum : fd数量
	*  flagList : 对应项如果有可以读的，对应位置置1 
	*  返回 0：表示超时
	*/
	FFLIB_API_IMPORT_EXPORT int32_t FFL_socketSelect(const NetFD *fdList, int8_t *flagList, size_t fdNum, int64_t timeoutUs);
#ifdef __cplusplus 
}
#endif


#endif

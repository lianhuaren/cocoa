 /*
 *  This file is part of FFL project.
 *
 *  The MIT License (MIT)
 *  Copyright (C) 2017-2018 zhufeifei All rights reserved.
 *
 *  FFL_NetClient.c
 *  Created by zhufeifei(34008081@qq.com) on 2017/8/12.
 *  https://github.com/zhenfei2016/FFLv2-lib.git
  *  连接指定服务器，参考android adb代码
 *
 */
#include "internalSocket.h"
#include <net/FFL_Net.h>

 /*
 *  如果fd指向的为NULL 则内部会进行socket的创建
 */



SOCKET_STATUS FFL_socketNetworkClientTimeout(const char *host, int port, int type, NetFD* fd, int timeout_tm)
{
    struct hostent *hp;
    struct sockaddr_in addr;
    int s;
	int error=0;

	if (inet_addr(host) != INADDR_NONE) {
		addr.sin_family = AF_INET;
		addr.sin_port = htons(port);
		addr.sin_addr.s_addr = inet_addr(host);
	}
	else {
		//gethostname(hostname, 100);   //获得主机的名称

		//getaddrinfo(host, NULL, &hints, &res);   //利用主机名称获取本地地址
		//char buff[100];
		//DWORD bufflen = 100;
		////将本地地址转换成字符串显示
		//struct sockaddr_in* pSockaddr = (sockaddr_in*)res->ai_addr;
		//char *pIP = inet_ntoa(pSockaddr->sin_addr);

		hp = gethostbyname(host);
		if (hp == 0) {
			return FFL_ERROR_SOCKET_GET_PEER_IP;
		}
		addr.sin_family = hp->h_addrtype;
		addr.sin_port = htons(port);
		memcpy(&addr.sin_addr, hp->h_addr, hp->h_length);
	}

	if (*fd == 0) {
		s = socket(addr.sin_family, type, 0);
		if (s < 0) {
			return FFL_ERROR_SOCKET_CREATE;
		}
	}
	else {
		s = *fd;
	}

    error=connect(s, (struct sockaddr *) &addr, sizeof(addr));
    if( error< 0){
        FFL_socketClose(s);
        return FFL_ERROR_SOCKET_CONNECT;
    }

	*fd = s;
    return FFL_SOCKET_OK;
}

SOCKET_STATUS FFL_socketNetworkClient(const char *host, int port, int type, NetFD*fd)
{
	return FFL_socketNetworkClientTimeout(host, port, type, fd, 0);
}


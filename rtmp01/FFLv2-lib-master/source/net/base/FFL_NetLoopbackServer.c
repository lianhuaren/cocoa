 /*
 *  This file is part of FFL project.
 *
 *  The MIT License (MIT)
 *  Copyright (C) 2017-2018 zhufeifei All rights reserved.
 *
 *  FFL_socketLoopbackServer.c
 *  Created by zhufeifei(34008081@qq.com) on 2017/8/12.
 *  https://github.com/zhenfei2016/FFLv2-lib.git
 *  当前机子的环回地址(127.0.0.1)创建服务器，参考android adb代码
 *
 */
#include "internalSocket.h"
#include <net/FFL_Net.h>

 /*
 *  如果fd指向的为NULL 则内部会进行socket的创建
 */
#define LISTEN_BACKLOG 4
SOCKET_STATUS FFL_socketLoopbackServer(int port, int type, NetFD *fd){
    struct sockaddr_in addr;
    int s, n;
	FFL_Zerop(&addr);
    
    addr.sin_family = AF_INET;
    addr.sin_port = htons(port);
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);

	if (*fd == 0) {
		s = socket(AF_INET, type, 0);
		if (s < 0) {
			return FFL_ERROR_SOCKET_CREATE;
		}
	}
	else {
		s = *fd;
	}

    n = 1;
    setsockopt(s, SOL_SOCKET, SO_REUSEADDR,(const char*) (&n), sizeof(n));
    if(bind(s, (struct sockaddr *) &addr, sizeof(addr)) < 0) {
        FFL_socketClose(s);
		return FFL_ERROR_SOCKET_BIND;
    }

    if (type == SOCK_STREAM) {
        int ret;
        ret = listen(s, LISTEN_BACKLOG);

        if (ret < 0) {
			FFL_socketClose(s);
            return FFL_ERROR_SOCKET_LISTEN;
        }
    }

	*fd = s;
    return FFL_SOCKET_OK;
}


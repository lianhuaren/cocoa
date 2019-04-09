/*
*  This file is part of FFL project.
*
*  The MIT License (MIT)
*  Copyright (C) 2017-2018 zhufeifei All rights reserved.
*
*  FFL_NetConst.h   
*  Created by zhufeifei(34008081@qq.com) on 2018/2/25. 
*  https://github.com/zhenfei2016/FFL-v2.git
*  网络socket错误码
*
*/
#ifndef _FFL_SOCKET_ERROR_H_
#define _FFL_SOCKET_ERROR_H_


#define FFL_SOCKET_OK FFL_OK
#define FFL_ERROR_SOCKET_SUCCESS                FFL_ERROR_SUCCESS
#define FFL_ERROR_SOCKET_CREATE                 1000
#define FFL_ERROR_SOCKET_SETREUSE               1001
#define FFL_ERROR_SOCKET_BIND                   1002
#define FFL_ERROR_SOCKET_LISTEN                 1003
#define FFL_ERROR_SOCKET_CLOSED                 1004
#define FFL_ERROR_SOCKET_GET_PEER_NAME          1005
#define FFL_ERROR_SOCKET_GET_PEER_IP            1006


#define FFL_ERROR_SOCKET_READ                   1007
#define FFL_ERROR_SOCKET_READ_FULLY             1008
#define FFL_ERROR_SOCKET_WRITE                  1009
#define FFL_ERROR_SOCKET_WAIT                   1010

/*  读失败，写失败误 */
#define FFL_ERROR_SOCKET_READ_EX                10000
#define FFL_ERROR_SOCKET_WRITE_EX               20000

/*  超时  */
#define FFL_ERROR_SOCKET_TIMEOUT                1011
#define FFL_ERROR_SOCKET_CONNECT                1012

#define  FFL_ERROR_SOCKET_FAILED                1013
/*  select 函数失败 */
#define FFL_ERROR_SOCKET_SELECT                -1000
/*
*  socket系列api,返回状态
*/
typedef int SOCKET_STATUS;




#endif

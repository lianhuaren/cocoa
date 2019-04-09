/*
 *  systhread.hpp
 *  FFL
 *
 *  Created by zhufeifei on 2017/11/12.
 *  Copyright (C) 2017-2018 zhufeifei All rights reserved.
 *  �����ⲿ��Ҫ��FFL_sys_Thread�ṹ�������ⲿ����ĳ�����
*/
#ifndef _STDCPP_SYS_THREAD_H_
#define _STDCPP_SYS_THREAD_H_

#include <FFL_atomic.h>
#include <thread/FFL_thread_constant.h>

typedef FFL_ThreadFunction FFL_ThreadLoop;
typedef void * SYS_ThreadHandle;
struct  FFL_sys_Thread
{
	int threadid;
	SYS_ThreadHandle handle;
	int status;
	FFL_atomic_t state;

	char *name;
	int stacksize;
	void *data;
};


#endif
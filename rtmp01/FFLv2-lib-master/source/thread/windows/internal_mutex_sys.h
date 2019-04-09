/*
 *  internal_mutex_sys.h
 *  FFL
 *
 *  Created by zhufeifei on 2017/11/12.
 *  Copyright (C) 2017-2018 zhufeifei All rights reserved.
 *  内部定义的mutex,信号量，条件结构，
*/
#ifndef _INTERNAL_MUTEX_SYS_H_
#define _INTERNAL_MUTEX_SYS_H_

#include <windows.h>

struct FFL_mutex_sys
{
	CRITICAL_SECTION cs;
};


struct FFL_semaphore_sys
{
	HANDLE id;
	LONG count;
};


#endif 

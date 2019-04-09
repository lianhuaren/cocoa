/*
 *  FFL_mutex_sys.h
 *  FFL
 *
 *  Created by zhufeifei on 2017/11/12.
 *  Copyright (C) 2017-2018 zhufeifei All rights reserved.
 *  通过引用这个头文件，加载真正的实现文件
*/
#ifndef _FFL_MUTEX_SYS_H_
#define _FFL_MUTEX_SYS_H_

#include "FFL_core.h"

#if FFL_THREADS_DISABLED
#error Need thread implementation for this platform
#elif FFL_THREAD_PTHREAD
#include "pthread/pthread_thread_internal.h"
#elif FFL_THREAD_WINDOWS
#include "windows/window_thread_internal.h"
#elif FFL_THREAD_STDCPP
#include "stdcpp/sysmutex.hpp"
#else
#error Need thread implementation for this platform
#endif


#endif 

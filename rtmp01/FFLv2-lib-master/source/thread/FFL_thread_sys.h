/*
 *  FFL_thread_sys.h
 *  FFL
 *
 *  Created by zhufeifei on 2017/11/12.
 *  Copyright (C) 2017-2018 zhufeifei All rights reserved.
 *  每个特定平台的线程需要实现的函数，FFL_thread.c会对这些实现进一步的包装
 *  每个特定实现的类，通过他可以应用外部定义的一些常量等
*/
#ifndef _FFL_SYS_THREAD_H_
#define _FFL_SYS_THREAD_H_

#include "FFL_Core.h"

#if FFL_THREADS_DISABLED
#error Need thread implementation for this platform
#elif FFL_THREAD_PTHREAD
#include "pthread/internal_thread_sys.h"
#elif FFL_THREAD_WINDOWS
#include "windows/internal_thread_sys.h"
#elif FFL_THREAD_STDCPP
#include "stdcpp/internal_thread_sys.h"
#else
#error Need thread implementation for this platform
#endif

#ifdef  __cplusplus
extern "C" {
#endif
	int FFL_SYS_CreateThread(struct FFL_Thread_sys * thread, void *args, FFL_ThreadLoop fn);

	/* 设置名称 */
	void FFL_SYS_SetThreadName(const char *name);

	/* 设置优先级 */
	int FFL_SYS_SetThreadPriority(FFL_ThreadPriority priority);

	/* 等待结束 */
	void FFL_SYS_WaitThread(struct FFL_Thread_sys * thread);

	/* 结束线程 */
	void FFL_SYS_DetachThread(struct FFL_Thread_sys * thread);

	/*当前线程id*/
	FFL_ThreadID FFL_SYS_ThreadID(void);

#ifdef  __cplusplus
}
#endif

#endif 

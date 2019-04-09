/*
 *  internal_thread_sys .h
 *  FFL
 *
 *  Created by zhufeifei on 2017/11/12.
 *  Copyright (C) 2017-2018 zhufeifei All rights reserved.
 *  内部线程实现文件 引用的线程结构体定义，
*/
#ifndef _INTERNAL_THREAD_SYS_H_
#define _INTERNAL_THREAD_SYS_H_

#ifdef  __cplusplus
extern "C" {
#endif
#include <FFL_Atomic.h>
#include <FFL_ThreadConstant.h>


#include <FFL_Core.h>

#if FFL_THREADS_DISABLED
#error Need thread implementation for this platform
#elif FFL_THREAD_PTHREAD

#elif FFL_THREAD_WINDOWS

#include <windows.h>
	typedef HANDLE SYS_ThreadHandle;

#elif FFL_THREAD_STDCPP

	typedef void* SYS_ThreadHandle;
#else
#error Need thread implementation for this platform
#endif


	typedef FFL_ThreadFunction FFL_ThreadLoop;


	struct FFL_Thread_sys
	{
		/*
		 * 线程id
		*/
		int threadid;

		/*
		*  线程句柄
		*/
		SYS_ThreadHandle handle;

		/*
		* 线程的状态，启动中，结束中....
		*/
		int status;
		AtomicInt state;

		/*
		*  线程名称
		*/
		char *name;


		int stacksize;
		void *data;
	};


#ifdef  __cplusplus
}
#endif
#endif 

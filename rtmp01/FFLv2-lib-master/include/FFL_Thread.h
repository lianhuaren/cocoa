/*
 *  FFL_thread.h
 *  FFL
 *
 *  Created by zhufeifei on 2017/8/12.
 *  Copyright (C) 2017-2018 zhufeifei All rights reserved.
 *  线程,主要移植于SDL库
 */

#ifndef _FFL_THREAD_H_
#define _FFL_THREAD_H_

#include <FFL_Core.h>
#include <FFL_ThreadConstant.h>

#ifdef  __cplusplus
extern "C" {
#endif

	struct FFL_Thread_sys;

	//前向声明的，cpp那面会进行真正的定义的
	//针对不同平台可能结构是不一样的
	//
	typedef struct FFL_Thread_sys FFL_Thread;


	//创建线程 并等待线程函数开始执行
	//当线程结束的时候，会自动删除创建的FFL_Thread
	// fn  : 线程函数
	//name ：线程名称
	//data : 上下文参数
	//
	FFLIB_API_IMPORT_EXPORT FFL_Thread *FFL_CALL FFL_CreateThread(FFL_ThreadFunction fn, const char *name, void *data);

	//获取线程名称
	//
	FFLIB_API_IMPORT_EXPORT const char *FFL_CALL  FFL_GetThreadName(FFL_Thread *thread);
	//获取线程id, 当前线程，其他线程
	//
	FFLIB_API_IMPORT_EXPORT FFL_ThreadID FFL_CALL  FFL_CurrentThreadID(void);
	FFLIB_API_IMPORT_EXPORT FFL_ThreadID FFL_CALL  FFL_GetThreadID(FFL_Thread * thread);
	//设置线程优先级
	//
	FFLIB_API_IMPORT_EXPORT int FFL_CALL  FFL_SetThreadPriority(FFL_ThreadPriority priority);
	/*
	 * 阻塞等待线程结束，返回后thread就不可用了
	*/
	FFLIB_API_IMPORT_EXPORT void FFL_CALL  FFL_WaitThread(FFL_Thread * thread, int *status);

	/*
	 * 结束线程  非阻塞，返回后线程还有资源没释放，等会会自己释放的，但是thread不可用了
	*/
	FFLIB_API_IMPORT_EXPORT void FFL_CALL  FFL_DetachThread(FFL_Thread * thread);

#ifdef  __cplusplus
}
#endif
#endif

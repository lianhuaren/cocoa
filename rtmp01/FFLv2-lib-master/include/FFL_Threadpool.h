/*
 *  FFL_threadpool.h
 *  FFL
 *
 *  Created by zhufeifei on 2017/8/12.
 *  Copyright (C) 2017-2018 zhufeifei All rights reserved.
 *  简单的线程池类
*/
#ifndef _FFL_THREADPOOL_H_
#define _FFL_THREADPOOL_H_

#include <FFL_Core.h>
#include <FFL_ThreadConstant.h>

#ifdef  __cplusplus
extern "C" {
#endif

	typedef struct sys_threadpool_t* FFL_Threadpool;
	typedef struct FFL_ThreadpoolJob_t
	{
		void *(FFL_CALL *func)(void *);
		void *arg;
	}FFL_ThreadpoolJob;

	/*
	 * 创建线程池  : threadnum线程数量
	 *
	 */
	FFLIB_API_IMPORT_EXPORT  FFL_Threadpool FFL_CALL FFL_CreateThreadpool(int threadnum);

	/*
	 * 运行一个job，如果当前没有可用的线程则等待
	 * 返回是否成功呢  FFL_ERROR_SUCCESS：成功
	 */
	FFLIB_API_IMPORT_EXPORT int FFL_CALL FFL_RunThreadpoolJob(FFL_Threadpool pool, FFL_ThreadpoolJob* job);

	/*
	 * 结束线程池
	 * 等待所有的线程结束后才返回
	 */
	FFLIB_API_IMPORT_EXPORT void FFL_CALL  FFL_DetachThreadpool(FFL_Threadpool  pool);


#ifdef  __cplusplus
}
#endif
#endif

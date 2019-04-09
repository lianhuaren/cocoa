/*
*  This file is part of FFL project.
*
*  The MIT License (MIT)
*  Copyright (C) 2017-2018 zhufeifei All rights reserved.
*
*  FFL_threadpool.c
*  Created by zhufeifei(34008081@qq.com) on 2017/08/12
*  https://github.com/zhenfei2016/FFLv2-lib.git
*  简单的线程池包装类
*
*/


#include <FFL_Threadpool.h>
#include "threadpool/threadpool.h"


/*  创建线程池 */
FFL_Threadpool FFL_CALL FFL_CreateThreadpool(int threadnum)
{
	sys_threadpool* pool=0;
	if(sys_threadpool_init(&pool,threadnum,0,0)==0)
	{
		return  pool;
	}

	return  0;
}

/*
 * 运行一个job，如果当前没有可用的线程则等待
 *
 */
int FFL_CALL FFL_RunThreadpoolJob(FFL_Threadpool pool,FFL_ThreadpoolJob* job)
{
	if(pool)
	{
		sys_threadpool_run(pool,job->func,job->arg);

		return FFL_ERROR_SUCCESS;
	}

	return FFL_ERROR_FAIL;
}


/*
 * 结束线程池
 * 等待所有的线程结束后zai返回
 */
void FFL_CALL  FFL_DetachThreadpool(FFL_Threadpool  pool)
{
	if(pool)
	{
		sys_threadpool_delete(pool);
	}
}



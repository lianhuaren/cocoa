/*
 *  systhreadpool.cpp
 *  FFL
 *
 *  Created by zhufeifei on 2017/8/12.
 *  Copyright (C) 2017-2018 zhufeifei All rights reserved.
 *  线程池
 */


#include "systhreadpool.hpp"

/*
 创建线程池
 */

extern "C"  FFL_sys_Thread_Pool*  FFL_SYS_CreateThreadPool(int threadnum)
{
	FFL_sys_Thread_Pool* pool = new FFL_sys_Thread_Pool();
	return pool;
}

/*
  等待线程池结束
  */
DECLSPEC void FFL_CALL  FFL_SYS_WaitThreadPool(FFL_sys_Thread_Pool pool, int *status)
{

}

/*
 结束线程池

 */
DECLSPEC void FFL_CALL  FFL_SYS_DetachThreadPool(FFL_sys_Thread_Pool  pool)
{

}

/*
 线程池中任务数量

 */
DECLSPEC int FFL_CALL FFL_SYS_ThreadPoolAddTask(FFL_sys_Thread_Pool pool, FFL_sys_Thread_Pool* task)
{
	return 0;
}

DECLSPEC int FFL_CALL FFL_SYS_WaitThreadPoolTask(FFL_sys_Thread_Pool pool, int taskid)
{
	return 0;
}

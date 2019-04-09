/*
 *  FFL_thread_constant.h
 *  FFL
 *
 *  Created by zhufeifei on 2017/8/12.
 *  Copyright (C) 2017-2018 zhufeifei All rights reserved.
 *  线程的一堆常量，单独分出来，因为线程的实现方需要这些值
 *
*/
#ifndef _FFL_THREAD_CONSTANT_H_
#define _FFL_THREAD_CONSTANT_H_


 /* 
    线程函数  
 
 */
typedef int (FFL_CALL * FFL_ThreadFunction) (void *data);

/* 
   线程id
   
*/
typedef unsigned long FFL_ThreadID;

/* 
  线程的优先级 
  
 */
typedef enum
{
	FFL_THREAD_PRIORITY_LOW,
	FFL_THREAD_PRIORITY_NORMAL,
	FFL_THREAD_PRIORITY_HIGH
} FFL_ThreadPriority;

/*
  线程状态 
  
  */
typedef enum FFL_ThreadState
{
	FFL_THREAD_STATE_ALIVE,
	FFL_THREAD_STATE_DETACHED,
	FFL_THREAD_STATE_ZOMBIE,
	FFL_THREAD_STATE_CLEANED,
} FFL_ThreadState;

#endif

 /*
 *  This file is part of FFL project.
 *
 *  The MIT License (MIT)
 *  Copyright (C) 2017-2018 zhufeifei All rights reserved.
 *
 *  FFL_Thread.c
 *  Created by zhufeifei(34008081@qq.com) on 2017/08/12
 *  https://github.com/zhenfei2016/FFLv2-lib.git
 *  线程,主要移植于SDL库
 *
 */

#include <FFL_Thread.h>
#include <FFL_Mutex.h>
#include "FFL_thread_sys.h"

typedef struct
{
    int (FFL_CALL * func) (void *);
    void *data;
    FFL_Thread *info;	
	FFL_sem *wait;
} thread_args;

/*  线程函数， 进行一些初始化操作的 */
static int FFL_CALL RunThread_loop(void *data)
{
    thread_args *args = (thread_args *) data;
    int ( FFL_CALL * userfunc) (void *) = args->func;
    void *userdata = args->data;
    FFL_Thread *thread = args->info;
    int *statusloc = &thread->status;
    
	//设置线程名称
	//
	FFL_SYS_SetThreadName(thread->name);

    //获取线程id
	//
    thread->threadid = FFL_SYS_ThreadID();

    //通知一下已经开始执行了
	//
	FFL_SemPost(args->wait);	

    /* 外部指定的线程函数 */
    *statusloc = userfunc(userdata);

    if (!FFL_atomicCmpxchg(&thread->state, FFL_THREAD_STATE_ALIVE, FFL_THREAD_STATE_ZOMBIE))
	{/*  如果结束前调用了 FFL_DetachThread,则进这里面.*/
        if (FFL_atomicCmpxchg(&thread->state, FFL_THREAD_STATE_DETACHED, FFL_THREAD_STATE_CLEANED))
		{
            if (thread->name)
			{
                FFL_free(thread->name);
                thread->name=0;
            }
			FFL_free(thread);
        }
    }
	return 0;
}


/*  创建一个线程，并且等待线程已经开始执行了 */
static FFL_Thread * FFL_CreateThreadWithStackSize(int ( FFL_CALL * fn) (void *),const char *name, const size_t stacksize, void *data)
{
    FFL_Thread *thread;
    thread_args *args;
    int ret;

    thread = (FFL_Thread *) FFL_mallocz(sizeof(*thread));
    if (thread == NULL) 
	{
        FFL_outofmemory();
        return (NULL);
    }

	thread->status=-1;
	FFL_atomicInit(&thread->state, FFL_THREAD_STATE_ALIVE);

    /*  保存线程名称 */
    if (name != NULL&&name[0]!=0)
	{
        thread->name = FFL_strdup(name);
        if (thread->name == NULL)
		{
			FFL_outofmemory();
            FFL_free(thread);
            return (NULL);
        }
    }

    /*  保存线程上线文参数 */
    args = (thread_args *) FFL_malloc(sizeof(*args));
    if (args == NULL)
	{
        FFL_outofmemory();
        if (thread->name)
		{
            FFL_free(thread->name);
            thread->name=0;
        }
        FFL_free(thread);
        return (NULL);
    }
    args->func = fn;
    args->data = data;
    args->info = thread;

    /* 创建信号，用于通知线程已经起来了 */
    args->wait = FFL_CreateSemaphore(0);
    if (args->wait == NULL)
	{
        if (thread->name)
		{
            FFL_free(thread->name);
        }
        FFL_free(thread);
        FFL_free(args);
        return (NULL);
    }
    thread->stacksize = stacksize;
	  
    ret = FFL_SYS_CreateThread(thread, args, RunThread_loop);
    if (ret >= 0)
	{   
        FFL_SemWait(args->wait);
    } else
	{
        if (thread->name)
		{
            FFL_free(thread->name);
        }
        FFL_free(thread);
        thread = NULL;
    }
	
	FFL_DestroySemaphore(args->wait);
    FFL_free(args);    
    return (thread);
}
/*
  对外的创建线程函数 

*/
 FFL_Thread * FFL_CALL FFL_CreateThread(FFL_ThreadFunction fn,const char *name, void *data)
{
    size_t stacksize = 0;
    return FFL_CreateThreadWithStackSize(fn, name, stacksize, data);
}

/* 获取线程名称  */
 const char * FFL_CALL FFL_GetThreadName(FFL_Thread * thread)
{
	if (thread)
	{
		return thread->name;
	}
	else
	{
		return NULL;
	}
}

 FFL_ThreadID FFL_CALL FFL_CurrentThreadID(void)
{
	return FFL_SYS_ThreadID();
}
/*获取线程*/
FFL_ThreadID FFL_GetThreadID(FFL_Thread * thread)
{
	FFL_ThreadID id;
    if (thread) 
	{
        id = thread->threadid;
    } else
	{
        id = FFL_CurrentThreadID();
    }
    return id;
}

/* 设置线程优先级 */
int FFL_SetThreadPriority(FFL_ThreadPriority priority)
{
    return FFL_SYS_SetThreadPriority(priority);
}

/*  等待线程结束 */
void FFL_WaitThread(FFL_Thread * thread, int *status)
{
    if (thread) 
	{
        FFL_SYS_WaitThread(thread);

        if (status) {
            *status = thread->status;
        }
        if (thread->name) {
            FFL_free(thread->name);
            thread->name=0;
        }
        FFL_free(thread);
    }
}

/* 结束线程 */
void FFL_DetachThread(FFL_Thread * thread)
{
    if (!thread)
	{
        return;
    }

    /* Grab dibs if the state is alive+joinable. */
    if (FFL_atomicCmpxchg(&thread->state, FFL_THREAD_STATE_ALIVE, FFL_THREAD_STATE_DETACHED))
	{
        FFL_SYS_DetachThread(thread);
    }
	else
	{
        /* all other states are pretty final, see where we landed. */
        const int thread_state = FFL_atomicValueGet(&thread->state);
        if ((thread_state == FFL_THREAD_STATE_DETACHED) || (thread_state == FFL_THREAD_STATE_CLEANED)) {
            return;  /* already detached (you shouldn't call this twice!) */
        } else if (thread_state == FFL_THREAD_STATE_ZOMBIE)
		{
			FFL_WaitThread(thread, NULL);  /* already done, clean it up. */
        } else
        {
			FFL_ASSERT_LOG(0 , "Unexpected thread state");
            return;
        }
    }
}

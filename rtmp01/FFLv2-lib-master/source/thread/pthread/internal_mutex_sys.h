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


#include <pthread.h>
#include <semaphore.h>

struct FFL_cond_sys
{
	pthread_cond_t cond;
};

typedef struct FFL_sys_cond SDL_cond;


//SDL_THREAD_PTHREAD_RECURSIVE_MUTEX_NP

#define  SDL_THREAD_PTHREAD_RECURSIVE_MUTEX 1

#if !SDL_THREAD_PTHREAD_RECURSIVE_MUTEX && \
    !SDL_THREAD_PTHREAD_RECURSIVE_MUTEX_NP
#define FAKE_RECURSIVE_MUTEX 1
#endif

struct FFL_mutex_sys
{
	pthread_mutex_t id;
#if FAKE_RECURSIVE_MUTEX
	int recursive;
	pthread_t owner;
#endif
};


#endif 

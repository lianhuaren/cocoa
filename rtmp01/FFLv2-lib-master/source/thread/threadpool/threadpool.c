/*****************************************************************************
 * threadpool.c: thread pooling
 *****************************************************************************
 * Copyright (C) 2010-2017 x264 project
 *
 * Authors: Steven Walters <kemuri9@gmail.com>
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02111, USA.
 *
 * This program is also available under a commercial proprietary license.
 * For more information, contact us at licensing@x264.com.
 *****************************************************************************/



/*
 *  threadpool.c
 *  FFL
 *  基于x264 中的线程池修改的,
 *  Created by zhufeifei on 2017/12/15.
 *  Copyright (C) 2017-2018 zhufeifei All rights reserved.
 *
*/
#include <FFL_Thread.h>
#include <FFL_Mutex.h>
#include "threadpool_job.h"
#include "threadpool.h"

typedef struct  sys_threadpool_t
{
    int            exit;
    int            threads;
    FFL_Thread ** thread_handle;
    void           (*init_func)(void *);
    void           *init_arg;

    /* requires a synchronized list structure and associated methods,
       so use what is already implemented for frames */
    FFL_sync_job_list uninit; /* list of jobs that are awaiting use */
    FFL_sync_job_list run;    /* list of jobs that are queued for processing by the pool */    
}sys_threadpool_t;

static int FFL_CALL sys_threadpoolhread( void *arg)
{
	sys_threadpool *pool = (sys_threadpool *)arg;
    if( pool->init_func )
        pool->init_func( pool->init_arg );

    while( !pool->exit )
    {
      
		FFL_threadpool_job *job = NULL;
        FFL_LockMutex( pool->run.mutex );
        while( !pool->exit && !pool->run.i_size )
            FFL_CondWait(pool->run.cv_fill,pool->run.mutex );

        if( pool->run.i_size )
        {
            job = FFL_job_list_pop_front_unlock( &(pool->run) );
            pool->run.i_size--;
        }
        FFL_UnlockMutex(pool->run.mutex );

        if( !job )
            continue;

        /* 执行*/
        job->ret = job->func( job->arg );

		/*  回收 */
        FFL_job_list_push( &pool->uninit, (void*)job );
    }
    return 0;
}

int sys_threadpool_init( sys_threadpool **p_pool, int threads,
                          void (*init_func)(void *), void *init_arg ){
	int i=0;
	sys_threadpool *pool=0;
    if( threads <= 0 )
        return -1;

    threads=FFL_MAX(threads,2);
    pool=FFL_mallocz(sizeof(sys_threadpool));
	if (pool == 0) {
		goto fail;
	}
    *p_pool = pool;

    pool->init_func = init_func;
    pool->init_arg  = init_arg;
    pool->threads   = threads;
	pool->thread_handle=FFL_mallocz( pool->threads * sizeof(FFL_Thread*) );
	if (pool->thread_handle == 0) {
		goto fail;
	}

    if( FFL_job_list_init( &pool->uninit, pool->threads ) ||
            FFL_job_list_init( &pool->run, pool->threads ))
        goto fail;

    for(  i = 0; i < pool->threads; i++ )
    {
       FFL_threadpool_job *job;
       job= FFL_mallocz(sizeof(FFL_threadpool_job));
	   if (job == NULL) {
		   return -1;
	   }
        FFL_job_list_push( &pool->uninit, job );
    }
    for(  i = 0; i < pool->threads; i++ ) {
        pool->thread_handle[i] = FFL_CreateThread(sys_threadpoolhread, "threadpool", pool);
        if (pool->thread_handle[i] == 0)
            goto fail;
    }

    return 0;
fail:
    return -1;
}

void sys_threadpool_run( sys_threadpool *pool, void *(*func)(void *), void *arg )
{
    FFL_threadpool_job *job = FFL_job_list_pop( &pool->uninit );
    job->func = func;
    job->arg  = arg;
    FFL_job_list_push( &pool->run,job );
}

void sys_threadpool_delete( sys_threadpool *pool ){
	int i=0;
    FFL_LockMutex( pool->run.mutex );
    pool->exit = 1;
    FFL_CondBroadcast( pool->run.cv_fill );
    FFL_UnlockMutex( pool->run.mutex );

    for( i = 0; i < pool->threads; i++ )
        FFL_WaitThread( pool->thread_handle[i], NULL );

    FFL_job_list_delete( &pool->uninit );
    FFL_job_list_delete( &pool->run );    
    FFL_free( pool->thread_handle );
    FFL_free( pool );
}

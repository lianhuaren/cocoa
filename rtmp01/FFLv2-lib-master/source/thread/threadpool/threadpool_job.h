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
 *  threadpool_job.h
 *  FFL
 *  基于x264 中的线程池修改的,这是线程池中的工作结构，和链表中读取，添加任务的接口
 *  Created by zhufeifei on 2017/12/15.
 *  Copyright (C) 2017-2018 zhufeifei All rights reserved.
 *
*/

#ifndef _FFL_THREAD_POOL_JOB_H_
#define _FFL_THREAD_POOL_JOB_H_


#include <FFL_Thread.h>
#include <FFL_Mutex.h>


typedef struct FFL_threadpool_job_t
{
    void *(*func)(void *);
    void *arg;
    void *ret;
} FFL_threadpool_job;


typedef struct FFL_sync_job_list_t
{
   FFL_threadpool_job **list;
   int i_max_size;
   int i_size;
   FFL_mutex*     mutex;
   FFL_cond*      cv_fill;  /* event signaling that the list became fuller */
   FFL_cond*     cv_empty; /* event signaling that the list became emptier */
} FFL_sync_job_list;

/*
 *   FFL_sync* 函数线程安全的，内部加锁了
 * */
int   FFL_job_list_init( FFL_sync_job_list *slist, int nelem );
void  FFL_job_list_delete( FFL_sync_job_list *slist );

void  FFL_job_list_push(FFL_sync_job_list *slist,FFL_threadpool_job* job);
FFL_threadpool_job *FFL_job_list_pop( FFL_sync_job_list *slist );

void  FFL_job_list_push_unlock(FFL_sync_job_list *slist, FFL_threadpool_job* job);
FFL_threadpool_job *FFL_job_list_pop_unlock(FFL_sync_job_list *slist);


/*
 *   返回头上第一条，不加锁的
 * */
FFL_threadpool_job *FFL_job_list_pop_front_unlock( FFL_sync_job_list *slist );




#endif

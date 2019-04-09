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
 *  threadpool_job.c
 *  FFL
 *  基于x264 中的线程池修改的,这是线程池中的工作结构，和链表中读取，添加任务的接口
 *  Created by zhufeifei on 2017/12/15.
 *  Copyright (C) 2017-2018 zhufeifei All rights reserved.
 *
*/

#include "threadpool_job.h"

int FFL_job_list_init( FFL_sync_job_list *slist, int max_size )
{
    if( max_size < 0 )
        return -1;
    slist->i_max_size = max_size;
    slist->i_size = 0;
    slist->list= FFL_mallocz ((max_size+1) * sizeof(FFL_threadpool_job*) );
	if (slist->list == 0) {
		return -1;
	}
    slist->mutex=FFL_CreateMutex();
    slist->cv_fill=FFL_CreateCond();
    slist->cv_empty=FFL_CreateCond();

    if( slist->mutex==0 ||
            slist->cv_fill==0 ||
            slist->cv_empty==0  )
    {
        FFL_DestroyMutex(slist->mutex);
        FFL_DestroyCond(slist->cv_fill);
        FFL_DestroyCond(slist->cv_empty);
        return -1;
    }
    return 0;
}

void FFL_job_list_delete( FFL_sync_job_list *slist )
{
    int i = 0;

    FFL_DestroyMutex( slist->mutex );
    FFL_DestroyCond( slist->cv_fill );
    FFL_DestroyCond( slist->cv_empty );

    if( !slist->list )
        return;
    while( slist->list[i] )
        FFL_free( slist->list[i++] );    
}

void FFL_job_list_push( FFL_sync_job_list *slist, FFL_threadpool_job *job )
{
    FFL_LockMutex( slist->mutex );
	FFL_job_list_push_unlock(slist, job);
	FFL_UnlockMutex(slist->mutex);
    
}

FFL_threadpool_job *FFL_job_list_pop( FFL_sync_job_list *slist )
{
    FFL_threadpool_job *job=0;
    FFL_LockMutex( slist->mutex );
	job=FFL_job_list_pop_unlock(slist);
    FFL_UnlockMutex( slist->mutex );
    return job;
}


void FFL_job_list_push_unlock(FFL_sync_job_list *slist, FFL_threadpool_job *job)
{
	while (slist->i_size == slist->i_max_size)
		FFL_CondWait(slist->cv_empty, slist->mutex);
	slist->list[slist->i_size++] = job;	
	FFL_CondBroadcast(slist->cv_fill);
}

FFL_threadpool_job *FFL_job_list_pop_unlock(FFL_sync_job_list *slist)
{
	FFL_threadpool_job *job;	
	while (!slist->i_size)
		FFL_CondWait(slist->cv_fill, slist->mutex);
	job = slist->list[--slist->i_size];
	slist->list[slist->i_size] = NULL;
	FFL_CondBroadcast(slist->cv_empty);	
	return job;
}

FFL_threadpool_job *FFL_job_list_pop_front_unlock( FFL_sync_job_list *slist )
{
    FFL_threadpool_job *job=slist->list[0];
    int i;
    for( i = 0; slist->list[i]; i++ )
        slist->list[i] = slist->list[i+1];
    assert(job);
    return job;
}

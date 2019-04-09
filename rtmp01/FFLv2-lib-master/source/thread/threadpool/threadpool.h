/*****************************************************************************
 * threadpool.h: thread pooling
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
*  This file is part of FFL project.
*
*  The MIT License (MIT)
*  Copyright (C) 2017-2018 zhufeifei All rights reserved.
*
*  threadpool_job
*  Created by zhufeifei(34008081@qq.com) on  2017/12/15.
*  基于x264 中的线程池修改的
*
*/


#ifndef _SYS_THREADPOOL_H_
#define _SYS_THREADPOOL_H_

typedef struct sys_threadpool_t sys_threadpool;
/*
 *    初始化线程池
 *    返回0表示成功
 *
 *    init_func:线程初始化时候执行的
 * */
int    sys_threadpool_init( sys_threadpool **p_pool, int threads, void (*init_func)(void *), void *init_arg );
void   sys_threadpool_run( sys_threadpool *pool, void *(*func)(void *), void *arg );
void*  sys_threadpool_wait( sys_threadpool *pool, void *arg );
void   sys_threadpool_delete( sys_threadpool *pool );

#endif

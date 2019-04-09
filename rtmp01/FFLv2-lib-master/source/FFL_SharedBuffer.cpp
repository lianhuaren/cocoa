/*
 * Copyright (C) 2005 The Android Open Source Project
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

 /*
 *  This file is part of FFL project.
 *
 *  The MIT License (MIT)
 *  Copyright (C) 2017-2018 zhufeifei All rights reserved.
 *
 *  FFL_SharedBuffer.cpp
 *  Created by zhufeifei(34008081@qq.com) on 2018/12/10
 *  https://github.com/zhenfei2016/FFL-v2.git
 *
 *  移植修改一些内容
 */

#include <stdlib.h>
#include <string.h>
#include <FFL_SharedBuffer.hpp>

#if WIN32
#include <windows.h>
#elif ANDROID
//#include <sys/atomics.h>
#include <stdatomic.h>
#elif MACOSX
//
//  废弃的老的atmoic系列函数  OSAtomic开头的
//
#include <libkern/OSAtomic.h>
//
//  c++11的，希望最新都用这个，但是变量需要 atomic_int  ,atomic_xx这样定义
//
#include <stdatomic.h>
#elif IOS
#include <libkern/OSAtomic.h>
#include <stdatomic.h>
#else
#include <FFL_Mutex.h>
static FFL_mutex* gAtomicMutex = NULL;
#endif

extern "C" void initializeSharedBuffer() {

}
extern "C" void terminateSharedBuffer() {

}
//_Atomic int
//
//  win32位下long跟int一样一样的
//
static void sharedBufferAtomicInc(volatile int32_t* v) {
#if WIN32
	InterlockedExchangeAdd((volatile unsigned long*)v, 1);
#elif ANDROID
	__sync_fetch_and_add(v, 1);
#elif MACOSX
	//
	//  这一系列的函数废弃了
	//
	OSAtomicAdd32(1, v);
#elif IOS
	OSAtomicAdd32(1, v);
#else

#endif
}
//
//  返回以前的值
//
static int32_t sharedBufferAtomicDec(volatile int32_t* v) {
#if WIN32
	return InterlockedExchangeAdd((volatile unsigned long*)v, -1);
#elif ANDROID
	return  __sync_fetch_and_sub(v, 1);
#elif MACOSX
	return OSAtomicDecrement32(v)+1;
#elif IOS
	return OSAtomicDecrement32(v)+1;
#else

#endif
}



namespace FFL {

SharedBuffer* SharedBuffer::alloc(size_t size)
{
    SharedBuffer* sb = static_cast<SharedBuffer *>(malloc(sizeof(SharedBuffer) + size));
    if (sb) {
        sb->mRefs = 1;
        sb->mSize = size;
    }
    return sb;
}


int SharedBuffer::dealloc(const SharedBuffer* released)
{
    if (released->mRefs != 0) return -1; // XXX: invalid operation
    free(const_cast<SharedBuffer*>(released));
    return 0;
}

SharedBuffer* SharedBuffer::edit() const
{
    if (onlyOwner()) {
        return const_cast<SharedBuffer*>(this);
    }
    SharedBuffer* sb = alloc(mSize);
    if (sb) {
        memcpy(sb->data(), data(), size());
        release();
    }
    return sb;    
}

SharedBuffer* SharedBuffer::editResize(size_t newSize) const
{
    if (onlyOwner()) {
        SharedBuffer* buf = const_cast<SharedBuffer*>(this);
        if (buf->mSize == newSize) return buf;
        buf = (SharedBuffer*)realloc(buf, sizeof(SharedBuffer) + newSize);
        if (buf != NULL) {
            buf->mSize = newSize;
            return buf;
        }
    }
    SharedBuffer* sb = alloc(newSize);
    if (sb) {
        const size_t mySize = mSize;
        memcpy(sb->data(), data(), newSize < mySize ? newSize : mySize);
        release();
    }
    return sb;    
}

SharedBuffer* SharedBuffer::attemptEdit() const
{
    if (onlyOwner()) {
        return const_cast<SharedBuffer*>(this);
    }
    return 0;
}

SharedBuffer* SharedBuffer::reset(size_t new_size) const
{
    // cheap-o-reset.
    SharedBuffer* sb = alloc(new_size);
    if (sb) {
        release();
    }
    return sb;
}

void SharedBuffer::acquire() const {
	sharedBufferAtomicInc(&mRefs);
}

int32_t SharedBuffer::release(uint32_t flags) const
{
    int32_t prev = 1;
    if (onlyOwner() || ((prev = sharedBufferAtomicDec(&mRefs)) == 1)) {
        mRefs = 0;
        if ((flags & eKeepStorage) == 0) {
            free(const_cast<SharedBuffer*>(this));
        }
    }
    return prev;
}




}; 

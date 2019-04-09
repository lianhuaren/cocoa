/*
 *  syscond.cpp
 *  FFL
 *
 *  Created by zhufeifei on 2017/8/12.
 *  Copyright (C) 2017-2018 zhufeifei All rights reserved.
 *  stl11库中的条件变量,主要移植于SDL库
 */



#include <FFL_Core.h>
#include <chrono>
#include <ratio>
#include <system_error>

#include "internal_mutex_sys.h"
#include <FFL_mutex.h>

extern "C" FFL_cond * FFL_CreateCond(void)
{    
    try {
        FFL_cond * cond = new FFL_cond();
        return cond;
    } catch (std::system_error & ex)
	{
        FFL_set_error("unable to create a C++ condition variable: code=%d; %s", (ex.code()), ex.what());
        return 0;
    } catch (std::bad_alloc &)
	{
        FFL_outofmemory();
        return 0;
    }
}

extern "C" void FFL_DestroyCond(FFL_cond * cond)
{
    if (cond) 
        delete cond;
}

/* Restart one of the threads that are waiting on the condition variable */
extern "C" int FFL_CondSignal(FFL_cond * cond)
{
    if (!cond) {
        FFL_set_error("Passed a NULL condition variable");
        return -1;
    }

    cond->cpp_cond.notify_one();
    return 0;
}

extern "C" int FFL_CondBroadcast(FFL_cond * cond)
{
    if (!cond) {
        FFL_set_error("Passed a NULL condition variable");
        return -1;
    }

    cond->cpp_cond.notify_all();
    return 0;
}

extern "C" int FFL_CondWaitTimeout(FFL_cond * cond, FFL_mutex * mutex, uint32_t ms)
{
    if (!cond) {
        FFL_set_error("Passed a NULL condition variable");
        return -1;
    }

    if (!mutex) {
        FFL_set_error("Passed a NULL mutex variable");
        return -1;
    }

    try {
        std::unique_lock<std::recursive_mutex> cpp_lock(mutex->cpp_mutex, std::adopt_lock_t());
        if (ms == FFL_MUTEX_MAXWAIT) {
            cond->cpp_cond.wait(
                cpp_lock
                );
            cpp_lock.release();
            return 0;
        } else {
            auto wait_result = cond->cpp_cond.wait_for(
                cpp_lock,
                std::chrono::duration<uint32_t, std::milli>(ms)
                );
            cpp_lock.release();
            if (wait_result == std::cv_status::timeout) {
				return 	ERROR_TIME_OUT ;
            } else {
                return 0;
            }
        }
    } catch (std::system_error & ex) {
        FFL_set_error("unable to wait on a C++ condition variable: code=%d; %s", (ex.code()), ex.what());
        return -1;
    }
}

/* Wait on the condition variable forever */
extern "C" int FFL_CondWait(FFL_cond * cond, FFL_mutex * mutex)
{
    return FFL_CondWaitTimeout(cond, mutex, FFL_MUTEX_MAXWAIT);
}

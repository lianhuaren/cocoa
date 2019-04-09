/*
 *  sysmutex.cpp
 *  FFL
 *
 *  Created by zhufeifei on 2017/8/12.
 *  Copyright (C) 2017-2018 zhufeifei All rights reserved.
 *  stl11库中的mutex,主要移植于SDL库
 */


#include <FFL_core.h>
#include <system_error>

#include "internal_mutex_sys.h"
#include <FFL_mutex.h>


/* Create a mutex */
extern "C" FFL_mutex * FFL_CreateMutex(void)
{    
    try {
        FFL_mutex * mutex = new FFL_mutex;
        return mutex;
    } catch ( std::system_error & ex) {
        FFL_set_error("unable to create a C++ mutex: code=%d; %s", (ex.code()), ex.what());
        return NULL;
    } catch (std::bad_alloc &) {
        FFL_outofmemory();
        return NULL;
    }
}

/* Free the mutex */
extern "C" void FFL_DestroyMutex(FFL_mutex * mutex)
{
    if (mutex) 
        delete mutex;
}

/* Lock the semaphore */
extern "C" int FFL_LockMutex(FFL_mutex * mutex)
{
    if (mutex == NULL) 
	{
        return FFL_set_error("Passed a NULL mutex");
    }

    try {
        mutex->cpp_mutex.lock();
        return FFL_ERROR_SUCCESS;
    } catch (std::system_error & ex) {
       return FFL_set_error("unable to lock a C++ mutex: code=%d; %s", ex.code(), ex.what());
    }
}

/* TryLock the mutex */
extern "C" int FFL_TryLockMutex(FFL_mutex * mutex)
{
    int retval = 0;
    if (mutex == NULL) 
	{
        return FFL_set_error("Passed a NULL mutex");
    }

    if (mutex->cpp_mutex.try_lock() == false) 
	{
        retval = ERROR_TIME_OUT;
    }
    return retval;
}

extern "C" int FFL_UnlockMutex(FFL_mutex * mutex)
{
    if (mutex == NULL) 
	{
        FFL_set_error("Passed a NULL mutex");
        return FFL_ERROR_FAIL;
    }

    mutex->cpp_mutex.unlock();
    return FFL_ERROR_SUCCESS;
}

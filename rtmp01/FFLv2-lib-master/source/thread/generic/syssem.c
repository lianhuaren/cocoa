/*
 *  syssem.c
 *  FFL
 *
 *  Created by zhufeifei on 2017/8/12.
 *  Copyright (C) 2017-2018 zhufeifei All rights reserved.
 *  通过mutex和条件变量模拟的一个信号,主要移植于SDL库
 */


#include <FFL_Core.h>
#include <FFL_Mutex.h>

#define FFL_MUTEX_MAXWAIT   (~(uint32_t)0)
void FFL_DestroySemaphore(FFL_sem * sem);

struct FFL_semaphore_sys
{
    uint32_t count;
	uint32_t waiters_count;
    FFL_mutex *count_lock;
    FFL_cond *count_nonzero;
};

FFL_sem * FFL_CreateSemaphore(uint32_t initial_value)
{
    FFL_sem *sem;

    sem = (FFL_sem *) FFL_malloc(sizeof(*sem));
    if (!sem) {
        FFL_outofmemory();
        return NULL;
    }
    sem->count = initial_value;
    sem->waiters_count = 0;

    sem->count_lock = FFL_CreateMutex();
    sem->count_nonzero = FFL_CreateCond();
    if (!sem->count_lock || !sem->count_nonzero) {
        FFL_DestroySemaphore(sem);
        return NULL;
    }

    return sem;
}

/* WARNING:
   You cannot call this function when another thread is using the semaphore.
*/
void FFL_DestroySemaphore(FFL_sem * sem)
{
    if (sem) {
        sem->count = 0xFFFFFFFF;
        while (sem->waiters_count > 0) {
            FFL_CondSignal(sem->count_nonzero);
            FFL_sleep(10);
        }
        FFL_DestroyCond(sem->count_nonzero);
        if (sem->count_lock) {
            FFL_LockMutex(sem->count_lock);
            FFL_UnlockMutex(sem->count_lock);
            FFL_DestroyMutex(sem->count_lock);
        }
        FFL_free(sem);
    }
}

int FFL_SemTryWait(FFL_sem * sem)
{
    int retval;

    if (!sem) {
        return FFL_set_error("Passed a NULL semaphore");
    }

    retval = FFL_MUTEX_MAXWAIT;
    FFL_LockMutex(sem->count_lock);
    if (sem->count > 0) {
        --sem->count;
        retval = 0;
    }
    FFL_UnlockMutex(sem->count_lock);

    return retval;
}

int FFL_SemWaitTimeout(FFL_sem * sem, uint32_t timeout)
{
    int retval;

    if (!sem) {
        return FFL_set_error("Passed a NULL semaphore");
    }

    /* A timeout of 0 is an easy case */
    if (timeout == 0) {
        return FFL_SemTryWait(sem);
    }

    FFL_LockMutex(sem->count_lock);
    ++sem->waiters_count;
    retval = 0;
    while ((sem->count == 0) && (retval != FFL_MUTEX_MAXWAIT)) {
        retval = FFL_CondWaitTimeout(sem->count_nonzero,
                                     sem->count_lock, timeout);
    }
    --sem->waiters_count;
    if (retval == 0) {
        --sem->count;
    }
    FFL_UnlockMutex(sem->count_lock);

    return retval;
}

int FFL_SemWait(FFL_sem * sem)
{
    return FFL_SemWaitTimeout(sem, FFL_MUTEX_MAXWAIT);
}

int32_t FFL_SemValue(FFL_sem * sem)
{
    int32_t value;

    value = 0;
    if (sem) {
        FFL_LockMutex(sem->count_lock);
        value = sem->count;
        FFL_UnlockMutex(sem->count_lock);
    }
    return value;
}

int FFL_SemPost(FFL_sem * sem)
{
    if (!sem) {
        return FFL_set_error("Passed a NULL semaphore");
    }

    FFL_LockMutex(sem->count_lock);
    if (sem->waiters_count > 0) {
        FFL_CondSignal(sem->count_nonzero);
    }
    ++sem->count;
    FFL_UnlockMutex(sem->count_lock);

    return 0;
}

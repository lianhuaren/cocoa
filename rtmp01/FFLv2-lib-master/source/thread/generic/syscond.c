#include "FFL_mutex.h"

struct FFL_cond_sys
{
    FFL_mutex *lock;
    int waiting;
    int signals;
    FFL_sem *wait_sem;
    FFL_sem *wait_done;
};

FFL_cond * FFL_CreateCond(void)
{
    FFL_cond *cond;
    cond = (FFL_cond *) FFL_mallocz(sizeof(FFL_cond));
    if (cond) 
	{
        cond->lock = FFL_CreateMutex();
        cond->wait_sem = FFL_CreateSemaphore(0);
        cond->wait_done = FFL_CreateSemaphore(0);
        cond->waiting = cond->signals = 0;
        if (!cond->lock || !cond->wait_sem || !cond->wait_done) 
		{
            FFL_DestroyCond(cond);
            cond = NULL;
        }
    } 
    return (cond);
}

void FFL_DestroyCond(FFL_cond * cond)
{
    if (cond) 
	{
        if (cond->wait_sem)
		{
            FFL_DestroySemaphore(cond->wait_sem);
        }

        if (cond->wait_done) 
		{
            FFL_DestroySemaphore(cond->wait_done);
        }

        if (cond->lock) 
		{
            FFL_DestroyMutex(cond->lock);
        }
        FFL_free(cond);
    }
}

/* Restart one of the threads that are waiting on the condition variable */
int FFL_CondSignal(FFL_cond * cond)
{
    if (!cond) {
        return FFL_set_error("Passed a NULL condition variable");
    }

    /* If there are waiting threads not already signalled, then
       signal the condition and wait for the thread to respond.
     */
    FFL_LockMutex(cond->lock);
    if (cond->waiting > cond->signals) {
        ++cond->signals;
        FFL_SemPost(cond->wait_sem);
        FFL_UnlockMutex(cond->lock);
        FFL_SemWait(cond->wait_done);
    } else {
        FFL_UnlockMutex(cond->lock);
    }

    return 0;
}

/* Restart all threads that are waiting on the condition variable */
int FFL_CondBroadcast(FFL_cond * cond)
{
    if (!cond) {
        return FFL_set_error("Passed a NULL condition variable");
    }

    /* If there are waiting threads not already signalled, then
       signal the condition and wait for the thread to respond.
     */
    FFL_LockMutex(cond->lock);
    if (cond->waiting > cond->signals) {
        int i, num_waiting;

        num_waiting = (cond->waiting - cond->signals);
        cond->signals = cond->waiting;
        for (i = 0; i < num_waiting; ++i) {
            FFL_SemPost(cond->wait_sem);
        }
        /* Now all released threads are blocked here, waiting for us.
           Collect them all (and win fabulous prizes!) :-)
         */
        FFL_UnlockMutex(cond->lock);
        for (i = 0; i < num_waiting; ++i) {
            FFL_SemWait(cond->wait_done);
        }
    } else {
        FFL_UnlockMutex(cond->lock);
    }

    return 0;
}

/* Wait on the condition variable for at most 'ms' milliseconds.
   The mutex must be locked before entering this function!
   The mutex is unlocked during the wait, and locked again after the wait.

Typical use:

Thread A:
    FFL_LockMutex(lock);
    while ( ! condition ) {
        SDL_CondWait(cond, lock);
    }
    FFL_UnlockMutex(lock);

Thread B:
    FFL_LockMutex(lock);
    ...
    condition = true;
    ...
    SDL_CondSignal(cond);
    FFL_UnlockMutex(lock);
 */
int FFL_CondWaitTimeout(FFL_cond * cond, FFL_mutex * mutex, uint32_t ms)
{
    int retval;

    if (!cond) {
        return FFL_set_error("Passed a NULL condition variable");
    }

    /* Obtain the protection mutex, and increment the number of waiters.
       This allows the signal mechanism to only perform a signal if there
       are waiting threads.
     */
    FFL_LockMutex(cond->lock);
    ++cond->waiting;
    FFL_UnlockMutex(cond->lock);

    /* Unlock the mutex, as is required by condition variable semantics */
    FFL_UnlockMutex(mutex);

    /* Wait for a signal */
    if (ms == FFL_MUTEX_MAXWAIT) {
        retval = FFL_SemWait(cond->wait_sem);
    } else {
        retval = FFL_SemWaitTimeout(cond->wait_sem, ms);
    }

    /* Let the signaler know we have completed the wait, otherwise
       the signaler can race ahead and get the condition semaphore
       if we are stopped between the mutex unlock and semaphore wait,
       giving a deadlock.  See the following URL for details:
       http://web.archive.org/web/20010914175514/http://www-classic.be.com/aboutbe/benewsletter/volume_III/Issue40.html#Workshop
     */
    FFL_LockMutex(cond->lock);
    if (cond->signals > 0) {
        /* If we timed out, we need to eat a condition signal */
        if (retval > 0) {
            FFL_SemWait(cond->wait_sem);
        }
        /* We always notify the signal thread that we are done */
        FFL_SemPost(cond->wait_done);

        /* Signal handshake complete */
        --cond->signals;
    }
    --cond->waiting;
    FFL_UnlockMutex(cond->lock);

    /* Lock the mutex, as is required by condition variable semantics */
    FFL_LockMutex(mutex);

    return retval;
}

/* Wait on the condition variable forever */
int FFL_CondWait(FFL_cond * cond, FFL_mutex * mutex)
{
    return FFL_CondWaitTimeout(cond, mutex, FFL_MUTEX_MAXWAIT);
}
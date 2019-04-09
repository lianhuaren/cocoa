/*
  Simple DirectMedia Layer
  Copyright (C) 1997-2016 Sam Lantinga <slouken@libsdl.org>

  This software is provided 'as-is', without any express or implied
  warranty.  In no event will the authors be held liable for any damages
  arising from the use of this software.

  Permission is granted to anyone to use this software for any purpose,
  including commercial applications, and to alter it and redistribute it
  freely, subject to the following restrictions:

  1. The origin of this software must not be misrepresented; you must not
     claim that you wrote the original software. If you use this software
     in a product, an acknowledgment in the product documentation would be
     appreciated but is not required.
  2. Altered source versions must be plainly marked as such, and must not be
     misrepresented as being the original software.
  3. This notice may not be removed or altered from any source distribution.
*/

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <errno.h>
#include <pthread.h>

#include <FFL_Core.h>
#include "internal_mutex_sys.h"
#include <FFL_Mutex.h>


FFL_mutex * FFL_CreateMutex(void)
{
    FFL_mutex *mutex;
    pthread_mutexattr_t attr;

    /* Allocate the structure */
    mutex = (FFL_mutex *) FFL_mallocz ( sizeof(*mutex));
    if (mutex)
    {
        pthread_mutexattr_init(&attr);
#if SDL_THREAD_PTHREAD_RECURSIVE_MUTEX
        pthread_mutexattr_settype(&attr, PTHREAD_MUTEX_RECURSIVE);
#elif SDL_THREAD_PTHREAD_RECURSIVE_MUTEX_NP
        pthread_mutexattr_setkind_np(&attr, PTHREAD_MUTEX_RECURSIVE_NP);
#else
        /* No extra attributes necessary */
#endif
        if (pthread_mutex_init(&mutex->id, &attr) != 0) {
            FFL_set_error("pthread_mutex_init() failed");

            FFL_free(mutex);
            mutex = NULL;
        }
    } else
    {
        FFL_outofmemory();
    }
    return (mutex);
}

void FFL_DestroyMutex(FFL_mutex * mutex)
{
    if (mutex) {
        pthread_mutex_destroy(&mutex->id);
        FFL_free(mutex);
    }
}

/* Lock the mutex */
int FFL_LockMutex(FFL_mutex * mutex)
{
#if FAKE_RECURSIVE_MUTEX
    pthread_t this_thread;
#endif

    if (mutex == NULL) {
        return FFL_set_error("Passed a NULL mutex");
    }

#if FAKE_RECURSIVE_MUTEX
    this_thread = pthread_self();
    if (mutex->owner == this_thread) {
        ++mutex->recursive;
    } else {
        /* The order of operations is important.
           We set the locking thread id after we obtain the lock
           so unlocks from other threads will fail.
         */
        if (pthread_mutex_lock(&mutex->id) == 0) {
            mutex->owner = this_thread;
            mutex->recursive = 0;
        } else {
            return FFL_set_error("pthread_mutex_lock() failed");
        }
    }
#else
    if (pthread_mutex_lock(&mutex->id) < 0) {
        return FFL_set_error("pthread_mutex_lock() failed");
    }
#endif
    return 0;
}

int FFL_TryLockMutex(FFL_mutex * mutex)
{
    int retval;
#if FAKE_RECURSIVE_MUTEX
    pthread_t this_thread;
#endif

    if (mutex == NULL) {
        return FFL_set_error("Passed a NULL mutex");
    }

    retval = 0;
#if FAKE_RECURSIVE_MUTEX
    this_thread = pthread_self();
    if (mutex->owner == this_thread) {
        ++mutex->recursive;
    } else {
        /* The order of operations is important.
         We set the locking thread id after we obtain the lock
         so unlocks from other threads will fail.
         */
        if (pthread_mutex_lock(&mutex->id) == 0) {
            mutex->owner = this_thread;
            mutex->recursive = 0;
        } else if (errno == EBUSY) {
            retval = ERROR_TIME_OUT;
        } else {
            retval = FFL_set_error("pthread_mutex_trylock() failed");
        }
    }
#else
    if (pthread_mutex_trylock(&mutex->id) != 0) {
        if (errno == EBUSY) {
            retval = FFL_MUTEX_TIMEDOUT;
        } else {
            retval = FFL_set_error("pthread_mutex_trylock() failed");
        }
    }
#endif
    return retval;
}

int FFL_UnlockMutex(FFL_mutex * mutex)
{
    if (mutex == NULL) {
        return FFL_set_error("Passed a NULL mutex");
    }

#if FAKE_RECURSIVE_MUTEX
    /* We can only unlock the mutex if we own it */
    if (pthread_self() == mutex->owner) {
        if (mutex->recursive) {
            --mutex->recursive;
        } else {
            /* The order of operations is important.
               First reset the owner so another thread doesn't lock
               the mutex and set the ownership before we reset it,
               then release the lock semaphore.
             */
            mutex->owner = 0;
            pthread_mutex_unlock(&mutex->id);
        }
    } else {
        return FFL_set_error("mutex not owned by this thread");
    }

#else
    if (pthread_mutex_unlock(&mutex->id) < 0) {
        return FFL_set_error("pthread_mutex_unlock() failed");
    }
#endif /* FAKE_RECURSIVE_MUTEX */

    return 0;
}


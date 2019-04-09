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

#include <sys/time.h>
#include <pthread.h>
#include <errno.h>

#include <FFL_Core.h>
#include "internal_mutex_sys.h"
#include <FFL_Mutex.h>


/* Create a condition variable */
FFL_cond *FFL_CreateCond(void)
{
    FFL_cond *cond;

    cond = (FFL_cond *) FFL_mallocz(sizeof(FFL_cond));
    if (cond) {
        if (pthread_cond_init(&cond->cond, NULL) < 0) {
            FFL_set_error("pthread_cond_init() failed");
            FFL_free(cond);
            cond = NULL;
        }
    }
    return (cond);
}

/* Destroy a condition variable */
void FFL_DestroyCond(FFL_cond * cond)
{
    if (cond) {
        pthread_cond_destroy(&cond->cond);
        FFL_free(cond);
    }
}

/* Restart one of the threads that are waiting on the condition variable */
int FFL_CondSignal(FFL_cond * cond)
{
    int retval;

    if (!cond) {
        return FFL_set_error("Passed a NULL condition variable");
    }

    retval = 0;
    if (pthread_cond_signal(&cond->cond) != 0) {
        return FFL_set_error("pthread_cond_signal() failed");
    }
    return retval;
}

/* Restart all threads that are waiting on the condition variable */
int FFL_CondBroadcast(FFL_cond * cond)
{
    int retval;

    if (!cond) {
        return FFL_set_error("Passed a NULL condition variable");
    }

    retval = 0;
    if (pthread_cond_broadcast(&cond->cond) != 0) {
        return FFL_set_error("pthread_cond_broadcast() failed");
    }
    return retval;
}

int FFL_CondWaitTimeout(FFL_cond * cond, FFL_mutex * mutex, uint32_t ms)
{
    int retval;
#ifndef HAVE_CLOCK_GETTIME
    struct timeval delta;
#endif
    struct timespec abstime;

    if (!cond) {
        return FFL_set_error("Passed a NULL condition variable");
    }

#ifdef HAVE_CLOCK_GETTIME
    clock_gettime(CLOCK_REALTIME, &abstime);

    abstime.tv_nsec += (ms % 1000) * 1000000;
    abstime.tv_sec += ms / 1000;
#else
    gettimeofday(&delta, NULL);

    abstime.tv_sec = delta.tv_sec + (ms / 1000);
    abstime.tv_nsec = (delta.tv_usec + (ms % 1000) * 1000) * 1000;
#endif
    if (abstime.tv_nsec > 1000000000) {
        abstime.tv_sec += 1;
        abstime.tv_nsec -= 1000000000;
    }

  tryagain:
    retval = pthread_cond_timedwait(&cond->cond, &mutex->id, &abstime);
    switch (retval) {
    case EINTR:
        goto tryagain;
        break;
    case ETIMEDOUT:
        retval = FFL_MUTEX_TIMEDOUT;
        break;
    case 0:
        break;
    default:
        retval = FFL_set_error("pthread_cond_timedwait() failed");
    }
    return retval;
}

/* Wait on the condition variable, unlocking the provided mutex.
   The mutex must be locked before entering this function!
 */
int FFL_CondWait(FFL_cond * cond, FFL_mutex * mutex)
{
    if (!cond) {
        return FFL_set_error("Passed a NULL condition variable");
    } else if (pthread_cond_wait(&cond->cond, &mutex->id) != 0) {
        return FFL_set_error("pthread_cond_wait() failed");
    }
    return 0;
}


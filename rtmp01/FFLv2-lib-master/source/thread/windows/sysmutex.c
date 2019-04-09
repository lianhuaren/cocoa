#include <windows.h>

#include <FFL_Core.h>
#include "internal_mutex_sys.h"
typedef struct FFL_mutex_sys  FFL_mutex;


FFL_mutex * FFL_CreateMutex(void)
{
    FFL_mutex *mutex;    
    mutex = (FFL_mutex *) FFL_mallocz(sizeof(*mutex));
    if (mutex) 
	{
        /* Initialize */
        /* On SMP systems, a non-zero spin count generally helps performance */
#if __WINRT__
        InitializeCriticalSectionEx(&mutex->cs, 2000, 0);
#else
        InitializeCriticalSectionAndSpinCount(&mutex->cs, 2000);
#endif
    }
	else 
	{
        FFL_outofmemory();
    }
    return (mutex);
}

void FFL_DestroyMutex(FFL_mutex * mutex)
{
    if (mutex) 
	{
        DeleteCriticalSection(&mutex->cs);
        FFL_free(mutex);
    }
}

int FFL_LockMutex(FFL_mutex * mutex)
{
    if (mutex == NULL)
	{
        return FFL_set_error("Passed a NULL mutex");
    }

    EnterCriticalSection(&mutex->cs);
    return (0);
}
int FFL_TryLockMutex(FFL_mutex * mutex)
{
    int retval = 0;
    if (mutex == NULL) 
	{
        return FFL_set_error("Passed a NULL mutex");
    }

    if (TryEnterCriticalSection(&mutex->cs) == 0)
	{
        retval = FFL_MUTEX_TIMEDOUT;
    }
    return retval;
}

int FFL_UnlockMutex(FFL_mutex * mutex)
{
    if (mutex == NULL)
	{
        return FFL_set_error("Passed a NULL mutex");
    }

    LeaveCriticalSection(&mutex->cs);
    return (0);
}

#include <windows.h>

#include <FFL_Core.h>
#include "internal_mutex_sys.h"
typedef struct FFL_semaphore_sys  FFL_sem;

/* Create a semaphore */
FFL_sem * FFL_CreateSemaphore(uint32_t initial_value)
{
    FFL_sem *sem;

    /* Allocate sem memory */
    sem = (FFL_sem *) FFL_mallocz(sizeof(*sem));
    if (sem) 
	{
        /* Create the semaphore, with max value 32K */
#if __WINRT__
        sem->id = CreateSemaphoreEx(NULL, initial_value, 32 * 1024, NULL, 0, SEMAPHORE_ALL_ACCESS);
#else
        sem->id = CreateSemaphore(NULL, initial_value, 32 * 1024, NULL);
#endif
        sem->count = initial_value;
        if (!sem->id) {
            FFL_set_error("Couldn't create semaphore");
            FFL_free(sem);
            sem = NULL;
        }
    } else {
        FFL_outofmemory();
    }
    return (sem);
}

/* Free the semaphore */
void FFL_DestroySemaphore(FFL_sem * sem)
{
    if (sem) {
        if (sem->id) {
            CloseHandle(sem->id);
            sem->id = 0;
        }
        FFL_free(sem);
    }
}

int FFL_SemWaitTimeout(FFL_sem * sem, int32_t timeout)
{
    int retval;
    DWORD dwMilliseconds;

    if (!sem) {
        return FFL_set_error("Passed a NULL sem");
    }

    if (timeout == FFL_MUTEX_MAXWAIT) {
        dwMilliseconds = INFINITE;
    } else {
        dwMilliseconds = (DWORD) timeout;
    }
#if __WINRT__
    switch (WaitForSingleObjectEx(sem->id, dwMilliseconds, FALSE)) {
#else
    switch (WaitForSingleObject(sem->id, dwMilliseconds)) {
#endif
    case WAIT_OBJECT_0:
        InterlockedDecrement(&sem->count);
        retval = 0;
        break;
    case WAIT_TIMEOUT:
        retval = FFL_MUTEX_TIMEDOUT;
        break;
    default:
        retval = FFL_set_error("WaitForSingleObject() failed");
        break;
    }
    return retval;
}

int FFL_SemWait(FFL_sem * sem)
{
	return FFL_SemWaitTimeout(sem, FFL_MUTEX_MAXWAIT);
}

int FFL_SemTryWait(FFL_sem * sem)
{
    return FFL_SemWaitTimeout(sem, 0);
}




int32_t FFL_SemValue(FFL_sem * sem)
{
    if (!sem) {
        FFL_set_error("Passed a NULL sem");
        return 0;
    }
    return (int32_t)sem->count;
}

int FFL_SemPost(FFL_sem * sem)
{
    if (!sem) {
        return FFL_set_error("Passed a NULL sem");
    }
    /* Increase the counter in the first place, because
     * after a successful release the semaphore may
     * immediately get destroyed by another thread which
     * is waiting for this semaphore.
     */
    InterlockedIncrement(&sem->count);
    if (ReleaseSemaphore(sem->id, 1, NULL) == FALSE) {
        InterlockedDecrement(&sem->count);      /* restore */
        return FFL_set_error("ReleaseSemaphore() failed");
    }
    return 0;
}


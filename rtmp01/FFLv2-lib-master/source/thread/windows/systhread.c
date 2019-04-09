#include <FFL_Core.h>
#include "internal_thread_sys.h"
typedef struct FFL_Thread_sys FFL_Thread;

//
//SDL_PASSED_BEGINTHREAD_ENDTHREAD  是否使用begin endthread进行创建线程
//
#ifndef SDL_PASSED_BEGINTHREAD_ENDTHREAD
/* We'll use the C library from this DLL */
#include <process.h>

#ifndef STACK_SIZE_PARAM_IS_A_RESERVATION
#define STACK_SIZE_PARAM_IS_A_RESERVATION 0x00010000
#endif
typedef uintptr_t(__cdecl * pfnSDL_CurrentBeginThread) (void *, unsigned,
                                                        unsigned (__stdcall *
                                                                  func) (void
                                                                         *),
                                                        void *arg, unsigned,
                                                        unsigned *threadID);
typedef void (__cdecl * pfnSDL_CurrentEndThread) (unsigned code);

#endif /* !SDL_PASSED_BEGINTHREAD_ENDTHREAD */

typedef struct ThreadStartParms
{
    void *args;
	FFL_ThreadLoop fn_loop;
    pfnSDL_CurrentEndThread pfnCurrentEndThread;
} tThreadStartParms, *pThreadStartParms;

static DWORD RunThread(void *data)
{
    pThreadStartParms pThreadParms = (pThreadStartParms) data;
    pfnSDL_CurrentEndThread pfnEndThread = pThreadParms->pfnCurrentEndThread;
    void *args = pThreadParms->args;
	FFL_ThreadLoop fn = pThreadParms->fn_loop;
    FFL_free(pThreadParms);

	if(fn)
	   fn(args);

    if (pfnEndThread != NULL)
        pfnEndThread(0);
    return (0);
}

static DWORD WINAPI RunThreadViaCreateThread(LPVOID data)
{
  return RunThread(data);
}

static unsigned __stdcall RunThreadViaBeginThreadEx(void *data)
{
  return (unsigned) RunThread(data);
}

#ifdef SDL_PASSED_BEGINTHREAD_ENDTHREAD
int internal_SYS_CreateThread(FFL_Thread * thread, FFL_ThreadLoop fn_loop, void *args,
                     pfnSDL_CurrentBeginThread pfnBeginThread,
                     pfnSDL_CurrentEndThread pfnEndThread)
{
#else
int internal_SYS_CreateThread(FFL_Thread * thread, FFL_ThreadLoop fn_loop, void *args)
{
    pfnSDL_CurrentBeginThread pfnBeginThread = (pfnSDL_CurrentBeginThread)_beginthreadex;
    pfnSDL_CurrentEndThread pfnEndThread = (pfnSDL_CurrentEndThread)_endthreadex;
#endif /* SDL_PASSED_BEGINTHREAD_ENDTHREAD */

    pThreadStartParms pThreadParms =
        (pThreadStartParms) FFL_mallocz(sizeof(tThreadStartParms));
    const DWORD flags = thread->stacksize ? STACK_SIZE_PARAM_IS_A_RESERVATION : 0;
    if (!pThreadParms)
	{
        return FFL_outofmemory();
    }

	
    /* Save the function which we will have to call to clear the RTL of calling app! */
    pThreadParms->pfnCurrentEndThread = pfnEndThread;
    
	/*  最终执行的线程函数和参数  */
	pThreadParms->fn_loop = fn_loop;
    pThreadParms->args = args;
    
    if (pfnBeginThread)
	{
        unsigned threadid = 0;
        thread->handle = (SYS_ThreadHandle)
            ((size_t) pfnBeginThread(NULL, (unsigned int) thread->stacksize,
                                     RunThreadViaBeginThreadEx,
                                     pThreadParms, flags, &threadid));
    } else 
	{
        DWORD threadid = 0;
        thread->handle = CreateThread(NULL, thread->stacksize,
                                      RunThreadViaCreateThread,
                                      pThreadParms, flags, &threadid);
    }
    if (thread->handle == NULL) 
	{
        return FFL_set_error("Not enough resources to create thread");
    }
    return 0;
}


int FFL_SYS_CreateThread(FFL_Thread * thread, void *args, FFL_ThreadLoop fn)
{
#ifdef SDL_PASSED_BEGINTHREAD_ENDTHREAD
	return internal_SYS_CreateThread(thread, fn, args,
		pfnSDL_CurrentBeginThread pfnBeginThread,
		pfnSDL_CurrentEndThread pfnEndThread);
#else
	return internal_SYS_CreateThread(thread, fn, args);
#endif
}

/*
*  设置线程名称
*/
#pragma pack(push,8)
typedef struct tagTHREADNAME_INFO
{
    DWORD dwType; /* must be 0x1000 */
    LPCSTR szName; /* pointer to name (in user addr space) */
    DWORD dwThreadID; /* thread ID (-1=caller thread) */
    DWORD dwFlags; /* reserved for future use, must be zero */
} THREADNAME_INFO;
#pragma pack(pop)

void FFL_SYS_SetThreadName(const char *name)
{
    if ((name != NULL) && IsDebuggerPresent()) {
        THREADNAME_INFO inf;

        ///* C# and friends will try to catch this Exception, let's avoid it. */
        //if (SDL_GetHintBoolean(SDL_HINT_WINDOWS_DISABLE_THREAD_NAMING, SDL_FALSE)) {
        //    return;
        //}

        /* This magic tells the debugger to name a thread if it's listening. */
        FFL_Zerop(&inf);
        inf.dwType = 0x1000;
        inf.szName = name;
        inf.dwThreadID = (DWORD) -1;
        inf.dwFlags = 0;

        /* The debugger catches this, renames the thread, continues on. */
        RaiseException(0x406D1388, 0, sizeof(inf) / sizeof(ULONG), (const ULONG_PTR*) &inf);
    }
}

FFL_ThreadID FFL_SYS_ThreadID(void)
{
    return ((FFL_ThreadID) GetCurrentThreadId());
}

int FFL_SYS_SetThreadPriority(FFL_ThreadPriority priority)
{
    int value;

    if (priority == FFL_THREAD_PRIORITY_LOW) {
        value = THREAD_PRIORITY_LOWEST;
    } else if (priority == FFL_THREAD_PRIORITY_HIGH) {
        value = THREAD_PRIORITY_HIGHEST;
    } else {
        value = THREAD_PRIORITY_NORMAL;
    }
    if (!SetThreadPriority(GetCurrentThread(), value)) {
        return FFL_set_error("SetThreadPriority()");
    }
    return 0;
}

void FFL_SYS_WaitThread(FFL_Thread * thread)
{
    WaitForSingleObjectEx(thread->handle, INFINITE, FALSE);
    CloseHandle(thread->handle);
}

void FFL_SYS_DetachThread(FFL_Thread * thread)
{
    CloseHandle(thread->handle);
}
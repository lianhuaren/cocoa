/*
 *  systhread.cpp
 *  FFL
 *
 *  Created by zhufeifei on 2017/8/12.
 *  Copyright (C) 2017-2018 zhufeifei All rights reserved.
 *  stl11库中的thread,主要移植于SDL库
 */
#include "internal_thread_sys.h"
typedef struct FFL_Thread_sys FFL_Thread;

#include <thread>
#include <system_error>
#ifdef WIN32
#include <windows.h>
#else
#include <sys/types.h>
#include <unistd.h>
#endif

static volatile int g_threadid=0;

typedef struct ThreadContext
{
	void *args;
	FFL_ThreadLoop fn;
}ThreadContext;
static void RunThread(void *args)
{
	ThreadContext* context = (ThreadContext*)args;
	//FFL_CPP_AUTOFREE(ThreadContext,context);
	(context->fn)(context->args);

	delete context;
}

extern "C" int FFL_SYS_CreateThread(FFL_Thread * thread, void *args,FFL_ThreadLoop fn)
{
	if (fn == 0) 
	{
		return FFL_ERROR_FAIL;
	}

	ThreadContext* context=0;	
    try {  		
		context= new ThreadContext();
		context->args = args;
		context->fn = fn;

        std::thread cpp_thread(RunThread, context);
        thread->handle = (void *) new std::thread(std::move(cpp_thread));
        return 0;
    } catch (std::system_error & ex) {
        FFL_set_error("unable to start a C++ thread: code=%d; %s", (ex.code()), ex.what());
		
        return -1;
    } catch (std::bad_alloc &) {
        FFL_outofmemory();
        return -1;
    }
}



extern "C" FFL_ThreadID FFL_SYS_ThreadID(void)
{
#ifdef WIN32
    return GetCurrentThreadId();
#elif ANDROID
    pid_t id= gettid();
    return id;
#else
    
    
    return g_threadid++;
#endif
}
extern "C" void FFL_SYS_SetThreadName(const char *name)
{
	FFL_SYS_ThreadID();
	return;
}

extern "C" int FFL_SYS_SetThreadPriority(FFL_ThreadPriority priority)
{
    // Thread priorities do not look to be settable via C++11's thread
    // interface, at least as of this writing (Nov 2012).  std::thread does
    // provide access to the OS' native handle, however, and some form of
    // priority-setting could, in theory, be done through this interface.
    //
    // WinRT: UPDATE (Aug 20, 2013): thread priorities cannot be changed
    // on WinRT, at least not for any thread that's already been created.
    // WinRT threads appear to be based off of the WinRT class,
    // ThreadPool, more info on which can be found at:
    // http://msdn.microsoft.com/en-us/library/windows/apps/windows.system.threading.threadpool.aspx
    //
    // For compatibility sake, 0 will be returned here.
    return (0);
}

extern "C" void FFL_SYS_WaitThread(FFL_Thread * thread)
{
    if ( ! thread) {
        return;
    }

    try {
        std::thread * cpp_thread = (std::thread *) thread->handle;
        if (cpp_thread->joinable()) {
            cpp_thread->join();
        }
    } catch (std::system_error &) {
        // An error occurred when joining the thread.  SDL_WaitThread does not,
        // however, seem to provide a means to report errors to its callers
        // though!
    }
}

extern "C" void FFL_SYS_DetachThread(FFL_Thread * thread)
{
    if ( ! thread) {
        return;
    }

    try {
        std::thread * cpp_thread = (std::thread *) thread->handle;
        if (cpp_thread->joinable()) {
            cpp_thread->detach();
        }
    } catch (std::system_error &) {
        // An error occurred when detaching the thread.  SDL_DetachThread does not,
        // however, seem to provide a means to report errors to its callers
        // though!
    }
}



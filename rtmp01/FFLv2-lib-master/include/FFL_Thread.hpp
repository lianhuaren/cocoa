/*
*  This file is part of FFL project.
*
*  The MIT License (MIT)
*  Copyright (C) 2017-2018 zhufeifei All rights reserved.
*
*  FFL_Thread   
*  Created by zhufeifei(34008081@qq.com) on 2017/11/25
*  https://github.com/zhenfei2016/FFLv2-lib.git
*  基予Android Open Source Project，修改的线程封装
*
*/

#ifndef _FFL_THREAD_HPP_
#define _FFL_THREAD_HPP_

#include <FFL_Core.h>
#include <FFL_Thread.h>
#include <FFL_Mutex.hpp>
#include <FFL_Ref.hpp>

namespace FFL {
	
class FFLIB_API_IMPORT_EXPORT Thread : virtual public RefBase
{
public:
            Thread();
    virtual ~Thread();

    // Start the thread in threadLoop() which needs to be implemented.
    virtual status_t  run(const char* name = 0,
                          int32_t priority = FFL_THREAD_PRIORITY_NORMAL,
                          size_t stack = 0);
    
    // Ask this object's thread to exit. This function is asynchronous, when the
    // function returns the thread might still be running. Of course, this
    // function can be called from a different thread.
    virtual void     requestExit();

    // Good place to do one-time initializations
    virtual status_t readyToRun();
    
    // Call requestExit() and wait until this object's thread exits.
    // BE VERY CAREFUL of deadlocks. In particular, it would be silly to call
    // this function from this object's thread. Will return WOULD_BLOCK in
    // that case.
     virtual status_t requestExitAndWait();

    // Wait until this object's thread exits. Returns immediately if not yet running.
    // Do not call from this object's thread; will return WOULD_BLOCK in that case.
			status_t    join();


			bool         isRunning() const;

			FFL_ThreadID getTid() const;

           // exit_pending() returns true if requestExit() has been called.
            bool        exitPending() const;
    
private:
	virtual void threadLoopStart();

    // Derived class must implement threadLoop(). The thread starts its life
    // here. There are two ways of using the Thread object:
    // 1) loop: if threadLoop() returns true, it will be called again if
    //          requestExit() wasn't called.
    // 2) once: if threadLoop() returns false, the thread will exit upon return.
    virtual bool threadLoop() = 0;

	virtual void threadLoopExit(const Thread* t);
private:    	
    static  int  FFL_CALL _threadLoop(void* user);
	class ThreadData;
	ThreadData* mThreadData;

	DISABLE_COPY_CONSTRUCTORS(Thread);
};

template class FFLIB_API_IMPORT_EXPORT FFL::sp<Thread>;

}; 

// ---------------------------------------------------------------------------
#endif // _LIBS_UTILS_THREAD_H
// ---------------------------------------------------------------------------

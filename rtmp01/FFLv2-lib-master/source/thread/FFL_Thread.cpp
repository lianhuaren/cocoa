/*
*  This file is part of FFL project.
*
*  The MIT License (MIT)
*  Copyright (C) 2017-2018 zhufeifei All rights reserved.
*
*  FFL_Thread.cpp
*  Created by zhufeifei(34008081@qq.com) on 2017/11/25
*  https://github.com/zhenfei2016/FFLv2-lib.git
*  基予Android Open Source Project，修改的线程封装
*
*/

#include <FFL_Thread.hpp>
#include "internalLogConfig.h"

namespace FFL {
	class Thread::ThreadData {
	public:
		ThreadData():mThread(0),
				   mStatus(FFL_NO_ERROR),
				   mExitPending(false), mRunning(false),
			       mTid(-1),mPriority(FFL_THREAD_PRIORITY_NORMAL)
		{				
		}

		~ThreadData(){
		}

		FFL_Thread*     mThread;
		mutable CMutex  mLock;
		CCondition      mThreadExitedCondition;
		CCondition      mThreadReadyCondition;
		status_t        mStatus;
		
		volatile bool           mExitPending;
		volatile bool           mRunning;
		sp<Thread>     mHoldSelf;

		FFL_ThreadID   mTid;
		FFL_ThreadPriority mPriority;

	};
	Thread::Thread():mThreadData(NULL){
		mThreadData = new ThreadData();
	}
	Thread::~Thread(){
		FFL_SafeFree(mThreadData);
	}
	status_t Thread::readyToRun(){
		return FFL_NO_ERROR;
	}

	status_t Thread::run(const char* name, int32_t priority, size_t stack){
		CMutex::Autolock _l(mThreadData->mLock);
		if (mThreadData->mRunning)
		{			
			return FFL_INVALID_OPERATION;
		}

		mThreadData->mPriority =(FFL_ThreadPriority) priority;
		// reset status and exitPending to their default value, so we can
		// try again after an error happened (either below, or in readyToRun())
		mThreadData->mStatus = FFL_NO_ERROR;
		mThreadData->mExitPending = false;

		// hold a strong reference on ourself
		mThreadData->mHoldSelf = this;
		mThreadData->mRunning = true;

		mThreadData->mThread = FFL_CreateThread(_threadLoop, name, this);
		if (mThreadData->mThread == NULL) {
			mThreadData->mStatus = FFL_ERROR_FAILED;   // something happened!
			mThreadData->mRunning = false;
			mThreadData->mHoldSelf.clear();  // "this" may have gone away after this.

			return FFL_ERROR_FAILED;
		}
		
		mThreadData->mThreadReadyCondition.signal();
		return FFL_NO_ERROR;
	}

	int Thread::_threadLoop(void* user)
	{
		Thread* const self = static_cast<Thread*>(user);

		sp<Thread> strong(self->mThreadData->mHoldSelf);
		wp<Thread> weak(strong);
		self->mThreadData->mHoldSelf.clear();
		bool first = true;
		bool exec_thread_exit = false;

		{//
		 //  等待run函数返回成功
		//
			CMutex::Autolock _l(self->mThreadData->mLock);
			if (self->mThreadData->mThread == NULL) {
				self->mThreadData->mThreadReadyCondition.wait(self->mThreadData->mLock);
			}
		}

		//
		//  保存一下线程名称
		//	
        unsigned long tid=self->mThreadData->mTid = FFL_CurrentThreadID();
		char threadName[256] = { 0 };
		const  char* tmpName = FFL_GetThreadName(self->mThreadData->mThread);
		if (tmpName != NULL && tmpName[0] != 0) {
			memcpy(threadName, tmpName, FFL_MIN(strlen(tmpName), 255));
		}
		INTERNAL_FFL_LOG_DEBUG("Thread(%d)(%s) run", tid, threadName);

		if (self->mThreadData->mPriority != FFL_THREAD_PRIORITY_NORMAL) {
			FFL_SetThreadPriority(self->mThreadData->mPriority);
		}


		self->threadLoopStart();		
		do {
			bool result;
			if (first) {
				first = false;
				self->mThreadData->mStatus = self->readyToRun();
				result = (self->mThreadData->mStatus == FFL_NO_ERROR);

				if (result && !self->exitPending()) {
					// Binder threads (and maybe others) rely on threadLoop
					// running at least once after a successful ::readyToRun()
					// (unless, of course, the thread has already been asked to exit
					// at that point).
					// This is because threads are essentially used like this:
					//   (new ThreadSubclass())->run();
					// The caller therefore does not retain a strong reference to
					// the thread and the thread would simply disappear after the
					// successful ::readyToRun() call instead of entering the
					// threadLoop at least once.
					result = self->threadLoop();
				}
			}
			else {
				result = self->threadLoop();
			}

			// establish a scope for mLock
			{
				CMutex::Autolock _l(self->mThreadData->mLock);
				if (result == false || self->mThreadData->mExitPending) {
					self->mThreadData->mExitPending = true;
					self->mThreadData->mRunning = false;
					
					// clear thread ID so that requestExitAndWait() does not exit if
					// called by a new thread using the same thread ID as this one.
					self->mThreadData->mThread = 0;
					
					self->threadLoopExit(self);
					exec_thread_exit = true;

					// note that interested observers blocked in requestExitAndWait are
					// awoken by broadcast, but blocked on mLock until break exits scope
					self->mThreadData->mThreadExitedCondition.broadcast();
					break;
				}
			}

			// Release our strong reference, to let a chance to the thread
			// to die a peaceful death.
			strong.clear();
			// And immediately, re-acquire a strong reference for the next loop
			strong = weak.promote();
		} while (!strong.is_empty());

		if(!exec_thread_exit)
		   self->threadLoopExit(0);

		INTERNAL_FFL_LOG_DEBUG("Thread(%d)(%s) exit", tid, threadName);
		return 0;
	}

	void Thread::requestExit()
	{
		CMutex::Autolock _l(mThreadData->mLock);
		mThreadData->mExitPending = true;
	}

	status_t Thread::requestExitAndWait()
	{
		CMutex::Autolock _l(mThreadData->mLock);
		if (mThreadData->mTid == FFL_CurrentThreadID()) {
			INTERNAL_FFL_LOG_WARNING(
				"Thread (this=%p): don't call waitForExit() from this "
				"Thread object's thread. It's a guaranteed deadlock!",
				this);

			return FFL_WOULD_BLOCK;
		}

		mThreadData->mExitPending = true;

		while (mThreadData->mRunning == true) {
			mThreadData->mThreadExitedCondition.wait(mThreadData->mLock);
		}
		// This next line is probably not needed any more, but is being left for
		// historical reference. Note that each interested party will clear flag.
		mThreadData->mExitPending = false;

		return mThreadData->mStatus;
	}

	status_t Thread::join()
	{
		CMutex::Autolock _l(mThreadData->mLock);
		if (mThreadData->mTid == FFL_CurrentThreadID()) {
			INTERNAL_FFL_LOG_WARNING(
				"Thread (this=%p): don't call join() from this "
				"Thread object's thread. It's a guaranteed deadlock!",
				this);

			return FFL_WOULD_BLOCK;
		}

		while (mThreadData->mRunning == true) {
			mThreadData->mThreadExitedCondition.wait(mThreadData->mLock);
		}

		return mThreadData->mStatus;
	}

	bool Thread::isRunning() const {
		CMutex::Autolock _l(mThreadData->mLock);
		return mThreadData->mRunning;
	}

	FFL_ThreadID Thread::getTid() const{	
		CMutex::Autolock _l(mThreadData->mLock);
		FFL_ThreadID tid;
		if (mThreadData->mRunning) {
			tid = FFL_GetThreadID(mThreadData->mThread);
		}
		else {
			INTERNAL_FFL_LOG_WARNING("Thread (this=%p): getTid() is undefined before run()", this);
			tid = -1;
		}
		return tid;
	}


	bool Thread::exitPending() const
	{
		CMutex::Autolock _l(mThreadData->mLock);
		return mThreadData->mExitPending;
	}

	void Thread::threadLoopStart()
	{
	}
	void Thread::threadLoopExit(const Thread* t)
	{
	}
}

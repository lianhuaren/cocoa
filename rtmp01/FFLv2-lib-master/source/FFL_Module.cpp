/*
*  This file is part of FFL project.
*
*  The MIT License (MIT)
*  Copyright (C) 2017-2018 zhufeifei All rights reserved.
*
*  FFL_Module.cpp
*  Created by zhufeifei(34008081@qq.com) on 2018/04/29
*  https://github.com/zhenfei2016/FFLv2-lib.git
*
*/

#include <FFL_Module.hpp>

namespace FFL {
	Module::Module():mStarted(false){
	}

	Module::~Module() {
	}
	//
	//  启动，
	//  thread： 使用使用这个线程进行eventloop
	//           =NULL , 需要外部调用eventLoop 
	//           !=NULL , 在这个thread中执行eventloop  
	//  返回是否启动成功
	//
	bool Module::start(FFL::sp<ModuleThread> thread) {
		CMutex::Autolock l(mLock);
		if (!mStarted) {
			if (!onStart()) {
				return false;
			}

			mStarted = true;
			mModuleThread = thread;			
			if (!mModuleThread.isEmpty()) {
				mModuleThread->setModule(this);
				mModuleThread->run();
			}
			return true;
		}
		return false;
	}
	//
	//   如果start使用了EventloopThread，则stop会阻塞到线程退出
	//   否则则仅仅置一下标志
	//
	bool Module::stop() {

		{
			CMutex::Autolock l(mLock);
			if (mStarted) {
				onStop();
				if (!mModuleThread.isEmpty()) {
					mModuleThread->requestExit();
				}

				mStarted = false;
			}
		}

		if (!mModuleThread.isEmpty()) {
			if (FFL_WOULD_BLOCK == mModuleThread->requestExitAndWait()) {
				return false;
			}
		}
		return true;
	}
	//
	//   等待线程退出
	//
	void Module::waitStop() {
		if (!mModuleThread.isEmpty()) {
			mModuleThread->requestExitAndWait();
		}
	}
	//
	//  调用。start，stop会触发onStart,onStop的执行
	//
	bool Module::onStart() {
		return true;
	}
	void Module::onStop() {

	}
	//
	//  是否启动状态
	//
	bool Module::isStarted() const{
		CMutex::Autolock l(mLock);
		return mStarted;
	}

	////////////////////////////////////////////////////////////////////////////////////////////////////
	//   跑module循环的线程
	////////////////////////////////////////////////////////////////////////////////////////////////////
    
    class ModuleThread::ThreadImpl : public Thread{
    public:
        ThreadImpl(ModuleThread* module){
            mModule=module;
        }
        
        bool threadLoop(){
            return  mModule->threadLoop();
        }
        ModuleThread* mModule;
    };
    
	ModuleThread::ModuleThread(const char* name):mName(NULL), mModule(NULL){
		if (name) {
			mName=FFL_strdup(name);
		}
		mModuleWaitMs = 0;
        mThread=new ThreadImpl(this);
	}
	ModuleThread::~ModuleThread() {
		if (mName) {
			FFL_free((void*)mName);
		}
	}
	//
	//   设置关联到哪一个module上
	//
	void ModuleThread::setModule(Module* module) {
		mModule = module;
	}
	//
	//  获取线程名称
	//
	const char* ModuleThread::getName() const {
		return mName?mName:"";
	}

    
    status_t  ModuleThread::run() {
		if (!mModule) {
			return FFL_ERROR_FAIL;
		}
		return mThread->run(mName);
	}
	void  ModuleThread::requestExit() {
		mCond.signal();
		mThread->requestExit();

	}
	status_t ModuleThread::requestExitAndWait() {
		mCond.signal();
		return mThread->requestExitAndWait();
	}

	/*  Thread  */
	bool ModuleThread::threadLoop() {
		if (mModuleWaitMs > 0) {
			mCond.waitRelative(mMutex, mModuleWaitMs);
		}

		if (mThread->exitPending()) {
			return false;
		}
		
		int32_t waitMs = 0;
		if (!mModule->eventLoop(&waitMs)) {
			return false;
		}
		mModuleWaitMs = waitMs;		
		return true;
	}
}

/*
*  This file is part of FFL project.
*
*  The MIT License (MIT)
*  Copyright (C) 2017-2018 zhufeifei All rights reserved.
*
*  FFL_Module.hpp
*  Created by zhufeifei(34008081@qq.com) on 2018/04/29
*  https://github.com/zhenfei2016/FFLv2-lib.git
*
*/
#ifndef _FFL_MODULE_HPP_
#define _FFL_MODULE_HPP_
#include <FFL_Mutex.hpp>
#include <FFL_Thread.hpp>

namespace FFL {
	class Module;
	////////////////////////////////////////////////////////////////////////////////////////////////////
	//   跑module循环的线程
	////////////////////////////////////////////////////////////////////////////////////////////////////
    class FFLIB_API_IMPORT_EXPORT ModuleThread :public RefBase{
		friend class Module;
	public:
		ModuleThread(const char* name);
		~ModuleThread();
		//
		//   设置关联到哪一个module上
		//
		void setModule(Module* module);
		//
		//  获取线程名称
		//
		const char* getName() const;
        //
        //  启动，停止线程
        //
        virtual status_t run();
        virtual void     requestExit();
		virtual status_t requestExitAndWait();
    protected:
		virtual bool threadLoop();
	private:
		const char* mName;
		Module* mModule;
		int32_t mModuleWaitMs;

		CMutex mMutex;
		CCondition mCond;
        
        class ThreadImpl;
        friend class ThreadImpl;
        FFL::sp<Thread> mThread;
	};
	template class FFLIB_API_IMPORT_EXPORT FFL::sp<FFL::ModuleThread>;

	////////////////////////////////////////////////////////////////////////////////////////////////////////
	class FFLIB_API_IMPORT_EXPORT Module {
	public:
		
	public:
		Module();
		virtual ~Module();

		//
		//  启动，
		//  thread： 使用使用这个线程进行eventloop
		//           =NULL , 需要外部调用eventLoop 
		//           !=NULL , 在这个thread中执行eventloop  
		//  返回是否启动成功
		//
		bool start(FFL::sp<ModuleThread> thread) ;
		//
		//   如果start使用了EventloopThread，则stop会阻塞到线程退出
		//   如果在自己线程中等待退出则不会阻塞的
		//   否则则仅仅置一下标志
		//   如果在自己线程中调用了stop，则返回false,可以在其他线程调用waitStop
		//
		bool stop() ;
		//
		//   等待线程退出
		//
		void waitStop();
	public:
		//
		//  调用。start，stop会触发onStart,onStop的执行
		//  onStart :表示准备开始了 ,可以做一些初始化工作
		//  onStop :表示准备停止了 ,可以做停止前的准备，想置一下信号让eventloop别卡住啊 
		//  在这些函数中，不要再调用自己的函数，例如：start,stop, isStarted等
		//
		virtual bool onStart();
		virtual void onStop();
		//
		//   阻塞的线程中执行的eventloop,返回是否继续进行eventLoop
		//   waitTime:输出参数，下一次执行eventLoop等待的时长
		//   true  : 继续进行下一次的eventLoop
		//   false : 不需要继续执行eventloop
		//
		virtual bool eventLoop(int32_t* waitTime) = 0;	
		//
		//  是否启动状态
		//
		bool isStarted() const;
	protected:
		volatile bool mStarted;
		mutable CMutex mLock;
		FFL::sp<ModuleThread> mModuleThread;
	};


}

#endif

/*
*  This file is part of FFL project.
*
*  The MIT License (MIT)
*  Copyright (C) 2017-2018 zhufeifei All rights reserved.
*
*  FFL_NetFdEvent.hpp   
*  Created by zhufeifei(34008081@qq.com) on 2018/12/1 
*  https://github.com/zhenfei2016/FFL-v2.git
*  网络读写的时间驱动类
*
*/
#ifndef _FFL_NET_FDEVENT_HPP_
#define _FFL_NET_FDEVENT_HPP_

#include <FFL_Core.h>
#include <FFL_Module.hpp>
#include <net/FFL_Net.h>

namespace FFL {
	class NetEventLoopImpl;
	class FFLIB_API_IMPORT_EXPORT NetEventLoop : public FFL::Module {
	public:
		class Callback {
		public:
			//
			//  返回是否还继续读写
			//  priv:透传数据
			//
			virtual bool onNetEvent(NetFD fd, bool readable, bool writeable, bool exception, void* priv) = 0;
		};
		typedef void(*CallbackFree)(Callback* h);
	public:
		//
		//  evenloopWaitUs:句柄多长时间轮训一次，默认0，一直轮训
		//   <0 ,一直等待到有数据
		//   >0  等待evenloopWaitUs毫秒
		//
		NetEventLoop(int64_t evenloopWaitUs=0);
		virtual ~NetEventLoop();
		//
		// 添加一个监听的句柄， readHandler如果不等NULL，则在removeFd的时候会调用 readHandlerFree进行清理
		// priv :透传到fdReady中
		// 
		bool addFd(NetFD fd,
			NetEventLoop::Callback* readHandler,
			NetEventLoop::CallbackFree readHandlerFree=NULL,
			void* priv=NULL);
		//
		//  移除这个句柄的处理handler
		//
		bool removeFd(NetFD fd);
	public:
		//////////  FFL::Module impl////////////////////////////////////////////////////////////////////////////////////
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
		virtual bool eventLoop(int32_t* waitTime);
	protected:
		NetEventLoopImpl* mImpl;
	};
}

#endif

/*
*  This file is part of FFL project.
*
*  The MIT License (MIT)
*  Copyright (C) 2017-2018 zhufeifei All rights reserved.
*
*  FFL_WebSocketServer.hpp
*  Created by zhufeifei(34008081@qq.com) on 2019/02/24
*  https://github.com/zhenfei2016/FFLv2-lib.git
*
*  websocket服务器
*/

#include <FFL_lib.hpp>


#ifndef _FFL_WEBSOCKET_SERVER_HPP_
#define _FFL_WEBSOCKET_SERVER_HPP_
namespace FFL {	
	class WebSocketServerImpl;
	class WebSocketClient;
	class FFLIB_API_IMPORT_EXPORT WebSocketServer : public Module {
	public:
		class Callback  {
		public:
			//
			//  建立删除一个websocket连接
			//
			virtual bool onClientCreate(WebSocketClient* fd) = 0;
			virtual void onClientDestroy(WebSocketClient* fd) = 0;
			//
			// 收到网络数据后，返回对这个client的处理方式   
			//			
			virtual bool onClientReceived(WebSocketClient* fd) = 0;
		};
	public:
		WebSocketServer(const char* ip, int32_t port, Callback* callback);
		virtual ~WebSocketServer();
	protected:
		//
		//  调用。start，stop会触发onStart,onStop的执行
		//  onStart :表示准备开始了 ,可以做一些初始化工作
		//  onStop :表示准备停止了 ,可以做停止前的准备，想置一下信号让eventloop别卡住啊 
		//  在这些函数中，不要再调用自己的函数，例如：start,stop, isStarted等
		//
		virtual bool onStart();
		virtual void onStop();
	public:
		//
		//   阻塞的线程中执行的eventloop,返回是否继续进行eventLoop
		//   waitTime:输出参数，下一次执行eventLoop等待的时长
		//   true  : 继续进行下一次的eventLoop
		//   false : 不需要继续执行eventloop
		//
		virtual bool eventLoop(int32_t* waitTime);
	private:
		WebSocketServerImpl* mImpl;
	};
}
#endif

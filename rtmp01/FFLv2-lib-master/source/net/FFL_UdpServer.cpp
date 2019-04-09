/*
*  This file is part of FFL project.
*
*  The MIT License (MIT)
*  Copyright (C) 2017-2018 zhufeifei All rights reserved.
*
*  FFL_UdpServer.cpp
*  Created by zhufeifei(34008081@qq.com) on 2018/07/14
*  https://github.com/zhenfei2016/FFLv2-lib.git
*  udp服务器
*
*/

#include <net/FFL_UdpServer.hpp>
#include <net/FFL_NetEventLoop.hpp>
#include <list>
#include <string>

namespace FFL {
	const static int64_t kEventLoopTimeout = 15000 * 1000;
	//
	//  udp缓冲区大小
	//
	const static int32_t KUdpBufferSize = 1024*10;
#define List std::list
	class UdpServerImpl {
	public:
		//
		//   ip:服务地址
		//   port: 服务端口
		//   handler ：处理句柄
		//   name:  服务名称
		//
		UdpServerImpl(const char* ip, uint16_t port,
			UdpServer::Callback* handler,
			const char* name = NULL);
		~UdpServerImpl();

		//
		//  调用。start，stop会触发onStart,onStop的执行
		//  onStart :表示准备开始了 ,可以做一些初始化工作
		//  onStop :表示准备停止了 ,可以做停止前的准备，想置一下信号让eventloop别卡住啊 
		//  在这些函数中，不要再调用自己的函数，例如：start,stop, isStarted等
		//
		bool onStart();
		void onStop();

		//
		//   阻塞的线程中执行的eventloop,返回是否继续进行eventLoop
		//   waitTime:输出参数，下一次执行eventLoop等待的时长
		//   true  : 继续进行下一次的eventLoop
		//   false : 不需要继续执行eventloop
		//
		bool eventLoop(int32_t* waitTime);
	protected:
		std::string mServerName;
		std::string mIP;
		uint16_t mPort;
		FFL::CSocket mServerSocket;
		
		bool onClientReceived(NetFD fd);
		UdpServer::Callback* mHandler;
	private:
		//
		//  监听socket是否有可读的消息了
		//
		class EventHandler : public NetEventLoop::Callback {
		public:
			EventHandler(UdpServerImpl* server) :mServer(server) {}
            virtual ~EventHandler(){}
			//
			//  返回是否还继续读写
			//  priv:透传数据
			//
			virtual bool onNetEvent(NetFD fd, bool readable, bool writeable, bool exception, void* priv);
			UdpServerImpl* mServer;
		};
		NetEventLoop* mEventLoop;
		EventHandler* mEventHandler;
	};

	UdpServerImpl::UdpServerImpl(const char* ip, uint16_t port,
		UdpServer::Callback* handler,
		const char* name):mHandler(handler){
		mServerName= name ? name : "";
		mIP = ip ? ip :"";
		mPort = port;
		mEventHandler = new EventHandler(this);
		mEventLoop = new NetEventLoop(kEventLoopTimeout);				
	}
	UdpServerImpl::~UdpServerImpl() {
		FFL_SafeFree(mEventLoop);
		FFL_SafeFree(mEventHandler);
	}
	//
	//  调用。start，stop会触发onStart,onStop的执行
	//  onStart :表示准备开始了 ,可以做一些初始化工作
	//  onStop :表示准备停止了 ,可以做停止前的准备，想置一下信号让eventloop别卡住啊 
	//  在这些函数中，不要再调用自己的函数，例如：start,stop, isStarted等
	//
	bool UdpServerImpl::onStart() {
		if (!mServerSocket.createUdpServer(mIP.c_str(),mPort)) {
			return false;
		}
		mEventLoop->addFd(mServerSocket.getFd(), mEventHandler, NULL,NULL );

        //
        //启动事件循环
        //
		bool ret= mEventLoop->start(NULL);
		if (!ret) {
			
		}
		return ret;
	}
	void UdpServerImpl::onStop() {		
		mEventLoop->stop();
	}
	//
	//   阻塞的线程中执行的eventloop,返回是否继续进行eventLoop
	//   waitTime:输出参数，下一次执行eventLoop等待的时长
	//   true  : 继续进行下一次的eventLoop
	//   false : 不需要继续执行eventloop
	//
	bool UdpServerImpl::eventLoop(int32_t* waitTime) {
		bool bContinue=mEventLoop->eventLoop(waitTime);		
		return bContinue;
	}



	bool UdpServerImpl::onClientReceived(NetFD fd) {
		uint8_t buf[KUdpBufferSize] = {};
		size_t size = 0;

		UdpClient client(fd);
		if (FFL_OK == mServerSocket.read(buf, KUdpBufferSize, &size)) {
			char srcIP[32] = {};
			uint16_t srcPort=0;
			mServerSocket.getReadFromAddr(srcIP, &srcPort);
			client.setWriteToAddr(srcIP, srcPort);
		}
		
		if (mHandler != NULL) {
			return mHandler->onClientReceived(&client,(const char*)buf,size);
		}
		return true;
	}
	//
	//  返回是否还继续读写
	//  priv:透传数据
	//
	bool UdpServerImpl::EventHandler::onNetEvent(NetFD fd, bool readable, bool writeable, bool exception, void* priv) {
        if (!mServer->onClientReceived(fd)) {			
			return true;
		}		
		return true;
	}
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	//   ip:服务地址
	//   port: 服务端口
	//   handler ：处理句柄
	//   name:  服务名称
	//
	UdpServer::UdpServer(const char* ip, uint16_t port,
		UdpServer::Callback* handler,
		const char* name ) {
		mImpl = new UdpServerImpl(ip, port, handler, name);
	}
	UdpServer::~UdpServer() {
		FFL_SafeFree(mImpl);
	}
	
	//
	//  调用。start，stop会触发onStart,onStop的执行
	//  onStart :表示准备开始了 ,可以做一些初始化工作
	//  onStop :表示准备停止了 ,可以做停止前的准备，想置一下信号让eventloop别卡住啊 
	//  在这些函数中，不要再调用自己的函数，例如：start,stop, isStarted等
	//
	bool UdpServer::onStart() {
		return mImpl->onStart();
	}
	void UdpServer::onStop() {
		return mImpl->onStop();
	}
	//
	//   阻塞的线程中执行的eventloop,返回是否继续进行eventLoop
	//   waitTime:输出参数，下一次执行eventLoop等待的时长
	//   true  : 继续进行下一次的eventLoop
	//   false : 不需要继续执行eventloop
	//
	bool UdpServer::eventLoop(int32_t* waitTime) {
		return mImpl->eventLoop(waitTime);
	}
}

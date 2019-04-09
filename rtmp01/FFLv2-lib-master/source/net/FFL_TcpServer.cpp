/*
*  This file is part of FFL project.
*
*  The MIT License (MIT)
*  Copyright (C) 2017-2018 zhufeifei All rights reserved.
*
*  FFL_TcpServer.cpp
*  Created by zhufeifei(34008081@qq.com) on 2018/07/14
*  https://github.com/zhenfei2016/FFLv2-lib.git
*  tcp服务器
*
*/

#include <net/FFL_TcpServer.hpp>
#include <list>
#include <string>
#include "internalLogConfig.h"

namespace FFL {
	// us   30s一次
	const static int64_t kEventLoopTimeout = 30000 * 1000 ;
#define List std::list
	class TcpServerImpl{
	public:
		//
		//   ip:服务地址
		//   port: 服务端口
		//   handler ：处理句柄
		//   name:  服务名称
		//
		TcpServerImpl(const char* ip, uint16_t port,
			TcpServer::Callback* handler,
			const char* name = NULL);
		~TcpServerImpl();	
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
	private:
		//
		//  处理超时
		//
		void processTimeout();
	protected:
		std::string mServerName;
		class ClientContext :public FFL::RefBase {
		public:
			ClientContext(NetFD fd);
			virtual ~ClientContext();

			NetFD mFd;
			TcpClient* mClient;
			//
			// 最后一次发送接受的时间点
			//
			int64_t mLastSendRecvTimeUs;
			//
			// 多长时间没有数据就关闭这个连接
			//
			int64_t mTimeoutUs;
		};

		CMutex mCLientsLock;
		std::list< FFL::sp<ClientContext> > mClients;
		void addClient(FFL::sp<ClientContext> contex);
		void removeClient(NetFD fd);

		//
		//  新连接上一个客户端
		//
		bool onClientCreated(ClientContext* context);
		//
		//  客户端关闭掉了
		//
		void onClientDestroyed(ClientContext* client, TcpServer::Callback::FD_OPTMODE mod);
		//
		//  有数据可以读了
		//
		TcpServer::Callback::FD_OPTMODE onClientReceived(ClientContext* context);


		TcpServer::Callback* mHandler;
	private:
		//
		//  监听一个客户端连接上来
		//
		class TcpListenerCb : public TcpListener::Callback {
		public:
			TcpListenerCb(TcpServerImpl* server);
            virtual ~TcpListenerCb(){}
			virtual void onAcceptClient(NetFD fd);
			TcpServerImpl* mServer;
		};
		TcpListener* mTcpListener;
		TcpListenerCb* mTcpListenerCallback;
	private:
		//
		//  监听socket是否有可读的消息了
		//
		class TcpEventHandler : public NetEventLoop::Callback {
		public:
			TcpEventHandler(TcpServerImpl* server) :mServer(server) {}
			//
			//  返回是否还继续读写
			//  priv:透传数据
			//
			virtual bool onNetEvent(NetFD fd, bool readable, bool writeable, bool exception, void* priv);
			TcpServerImpl* mServer;
		};
		NetEventLoop* mEventLoop;
		TcpEventHandler* mEventHandler;
	};	

	TcpServerImpl::TcpServerImpl(const char* ip, uint16_t port,
		                 TcpServer::Callback* handler,
		                 const char* name):mHandler(handler){
		FFL_socketInit();
		mTcpListenerCallback = new TcpListenerCb(this);
        mTcpListener = new TcpListener(ip,port, mTcpListenerCallback);

		mEventHandler = new TcpEventHandler(this);
		mEventLoop = new NetEventLoop(kEventLoopTimeout);		
	}
	TcpServerImpl::~TcpServerImpl() {
		FFL_SafeFree(mTcpListener);
		FFL_SafeFree(mTcpListenerCallback);

		FFL_SafeFree(mEventLoop);
		FFL_SafeFree(mEventLoop);
	}



	//
	//  调用。start，stop会触发onStart,onStop的执行
	//  onStart :表示准备开始了 ,可以做一些初始化工作
	//  onStop :表示准备停止了 ,可以做停止前的准备，想置一下信号让eventloop别卡住啊 
	//  在这些函数中，不要再调用自己的函数，例如：start,stop, isStarted等
	//
	bool TcpServerImpl::onStart() {
        //
        //  启动服务器监听,在一个单独的线程中运行
        //
		std::string name = mServerName;
		name += "_listener";
		if (!mTcpListener->start(new ModuleThread(name.c_str()))) {
			INTERNAL_FFL_LOG_WARNING("Failed to start tcp server.");
			return false;
		}

        //
        //启动事件循环
        //
		bool ret= mEventLoop->start(NULL);
		if (!ret) {
			mTcpListener->stop();
		}
		return ret;
	}
	void TcpServerImpl::onStop() {
		mTcpListener->stop();
		mEventLoop->stop();
	}
	//
	//   阻塞的线程中执行的eventloop,返回是否继续进行eventLoop
	//   waitTime:输出参数，下一次执行eventLoop等待的时长
	//   true  : 继续进行下一次的eventLoop
	//   false : 不需要继续执行eventloop
	//
	bool TcpServerImpl::eventLoop(int32_t* waitTime) {
		bool bContinue=mEventLoop->eventLoop(waitTime);
		processTimeout();
		return bContinue;
	}
	//
	//  处理超时
	//
	void TcpServerImpl::processTimeout() {
		int64_t now = FFL_getNowUs();
		std::list< FFL::sp<ClientContext> > timeoutClients;
		{
			FFL::CMutex::Autolock l(mCLientsLock);

			std::list< FFL::sp<ClientContext> >::iterator it = mClients.begin();
			for (; it != mClients.end(); it++) {
				FFL::sp<ClientContext> context = *it;

				if (context->mTimeoutUs < 0) {
					continue;
				}
				if (now > context->mLastSendRecvTimeUs) {
					if (now - context->mLastSendRecvTimeUs > context->mTimeoutUs) {
						timeoutClients.push_back(*it);
					}
				}
			}
		}

		if (timeoutClients.size() == 0) {
			return;
		}

		 //
		 //  超时
		 //
		std::list< FFL::sp<ClientContext> >::iterator it = timeoutClients.begin();
		for (; it != timeoutClients.end(); it++) {
			FFL::sp<ClientContext> context = *it;
            onClientDestroyed(context.get(), TcpServer::Callback::FD_DESTROY);
		}
	}
	TcpServerImpl::ClientContext::ClientContext(NetFD fd) : mFd(fd){
		mLastSendRecvTimeUs = FFL_getNowUs();
		//
		// 30秒没有数据就关闭
		//
		mTimeoutUs = 30 * 1000 * 1000;
		mClient = new TcpClient(fd);
	}
	TcpServerImpl::ClientContext::~ClientContext() {
		FFL_SafeFree(mClient);
	}
	//
	//  增加到监控列表
	//
	void TcpServerImpl::addClient(FFL::sp<ClientContext> contex) {
		FFL::CMutex::Autolock l(mCLientsLock);
		mClients.push_back(contex);
	}

	//
	//  移除这个fd，不监控它是否可读
	//
	void TcpServerImpl::removeClient(NetFD fd) {
		FFL::CMutex::Autolock l(mCLientsLock);
		std::list< FFL::sp<ClientContext> >::iterator it = mClients.begin();
		for (; it != mClients.end(); it++) {
			FFL::sp<ClientContext> context = *it;
			if (context->mFd == fd) {				
				mClients.erase(it);
				return;
			}
		}
	}
	//
	//  新连接上一个客户端
	//
	bool TcpServerImpl::onClientCreated(ClientContext* context) {		
		bool ret = false;
		if (context) {
			addClient(context);			
			if (mHandler != NULL) {
				ret= mHandler->onClientCreate(context->mClient,&context->mTimeoutUs);
			}

			if (!ret || !mEventLoop->addFd(context->mFd, mEventHandler, NULL, context)) {
				onClientDestroyed(context, TcpServer::Callback::FD_DESTROY);
			}
		}
		return ret;
	}
	//
	//  客户端关闭掉了
	//
	void TcpServerImpl::onClientDestroyed(ClientContext* context, TcpServer::Callback::FD_OPTMODE mod) {
		if (context) {
			mEventLoop->removeFd(context->mFd);
			if (mHandler != NULL) {
				mHandler->onClientDestroy(context->mClient,mod);
			}

			//
			//是否需要关闭这个连接 ,如果外部需要使用这个句柄怎么处理呢
			//
			if (mod == TcpServer::Callback::FD_DESTROY) {
				context->mClient->close();
			}else {
				context->mClient = NULL;
			}
			removeClient(context->mFd);
		}
	}
	//
	//  有数据可以读了
	//
	TcpServer::Callback::FD_OPTMODE TcpServerImpl::onClientReceived(ClientContext* context) {
		if (mHandler != NULL) {
			return mHandler->onClientReceived(context->mClient);
		}

		//
		//  关闭
		return TcpServer::Callback::FD_DESTROY;
	}
	TcpServerImpl::TcpListenerCb::TcpListenerCb(TcpServerImpl* server) :mServer(server) {
	}
	void TcpServerImpl::TcpListenerCb::onAcceptClient(NetFD fd) {
		if (fd != INVALID_NetFD) {
			FFL::sp<ClientContext> contex=new ClientContext(fd);
			if (mServer->onClientCreated(contex.get())) {
				
			}
		}
	}
	//
	//  返回是否还继续读写
	//  priv:透传数据
	//
	bool TcpServerImpl::TcpEventHandler::onNetEvent(NetFD fd, bool readable, bool writeable, bool exception, void* priv) {
		ClientContext* context = (ClientContext*)priv;
		TcpServer::Callback::FD_OPTMODE mod = mServer->onClientReceived(context);		
		//
		//  是否继续下一次的监控
		//
		if(mod!= TcpServer::Callback::FD_CONTINUE){
			mServer->onClientDestroyed(context, mod);
			return false;
		}		
		return true;
	}


	//
	//   ip:服务地址
	//   port: 服务端口
	//   handler ：处理句柄
	//   name:  服务名称
	//
	TcpServer::TcpServer(const char* ip, uint16_t port,
		TcpServer::Callback* handler,
		const char* name ) {
		mImpl = new TcpServerImpl(ip, port, handler, name);
	}
	TcpServer::~TcpServer(){
		FFL_SafeFree(mImpl);
	}
	//
	//   阻塞的线程中执行的eventloop,返回是否继续进行eventLoop
	//   waitTime:输出参数，下一次执行eventLoop等待的时长
	//   true  : 继续进行下一次的eventLoop
	//   false : 不需要继续执行eventloop
	//
	bool TcpServer::eventLoop(int32_t* waitTime) {
		return mImpl->eventLoop(waitTime);
	}
	bool TcpServer::onStart() {
		return mImpl->onStart();
	}
	void TcpServer::onStop() {
		mImpl->onStop();
	}
}

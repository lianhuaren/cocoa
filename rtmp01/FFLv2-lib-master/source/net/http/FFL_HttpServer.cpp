/*
*  This file is part of FFL project.
*
*  The MIT License (MIT)
*  Copyright (C) 2017-2018 zhufeifei All rights reserved.
*
*  FFL_HttpServer.cpp
*  Created by zhufeifei(34008081@qq.com) on 2018/11/08
*  https://github.com/zhenfei2016/FFLv2-lib.git
*
*  http服务器
*/

#include <net/http/FFL_HttpServer.hpp>
#include <net/http/FFL_HttpRequest.hpp>
#include <net/http/FFL_HttpResponse.hpp>
#include <net/http/FFL_HttpUrl.hpp>
#include <net/FFL_TcpClient.hpp>
#include <net/FFL_TcpServer.hpp>
#include <net/http/FFL_HttpClient.hpp>
#include <list>

namespace FFL {
	//
	//  具体服务器实现
	//
	class HttpServerImpl :public TcpServer::Callback{		
		friend class HttpServer;
		public:
			HttpServerImpl(const char* ip, int32_t port);
			virtual ~HttpServerImpl();
		public:		
			//
			//  注册处理指定http ，请求的处理句柄
			//
			void registerApi(const HttpServer::HttpApiKey& key, FFL::sp<HttpServer::Callback> handler);
			FFL::sp<HttpServer::Callback> unregisterApi(const HttpServer::HttpApiKey& key);
			FFL::sp<HttpServer::Callback> getRegisterApi(const HttpServer::HttpApiKey& key);

	
			struct ApiEntry {
				HttpServer::HttpApiKey mKey;
				FFL::sp<HttpServer::Callback> mHandler;
			};
			std::list<ApiEntry*> mApiList;
			CMutex mApiListLock;
	    protected:
			//
			//  处理http请求
			//
			bool processHttpRequest(FFL::sp<HttpRequest> request);	
			//
			//  调用。start，stop会触发onStart,onStop的执行
			//  onStart :表示准备开始了 ,可以做一些初始化工作
			//  onStop :表示准备停止了 ,可以做停止前的准备，想置一下信号让eventloop别卡住啊 
			//  在这些函数中，不要再调用自己的函数，例如：start,stop, isStarted等
			//
			FFL_INLINE bool onStart();
			FFL_INLINE void onStop();
			//
			//   阻塞的线程中执行的eventloop,返回是否继续进行eventLoop
			//   waitTime:输出参数，下一次执行eventLoop等待的时长
			//   true  : 继续进行下一次的eventLoop
			//   false : 不需要继续执行eventloop
			//
			FFL_INLINE bool eventLoop(int32_t* waitTime);
		private:	
			// class TcpServer::Callback
			//
			//  aliveTimeUs:保活时长，如果超过这么长时间还没有数据则干掉这个client
			//              <0 一直存活， 
			//
			virtual bool onClientCreate(TcpClient* client, int64_t* aliveTimeUs);
			virtual void onClientDestroy(TcpClient* client, TcpServer::Callback::FD_OPTMODE reason);
			virtual TcpServer::Callback::FD_OPTMODE onClientReceived(TcpClient* client);
	    private:
			TcpServer* mTcpServer;			
	};

	HttpServerImpl::HttpServerImpl(const char* ip, int32_t port){		
		mTcpServer = new TcpServer(ip,port, this);
	}
	HttpServerImpl::~HttpServerImpl(){
		FFL_SafeFree(mTcpServer);		
	}
	//
	//  注册处理指定http ，请求的处理句柄
	//
	void HttpServerImpl::registerApi(const HttpServer::HttpApiKey& key, FFL::sp<HttpServer::Callback> handler) {
		if (handler == NULL) {
			return;
		}

		ApiEntry* entry = new ApiEntry();
		entry->mKey = key;
		entry->mHandler = handler;

		FFL::CMutex::Autolock l(mApiListLock);
		mApiList.push_back(entry);
	}
	FFL::sp<HttpServer::Callback> HttpServerImpl::unregisterApi(const HttpServer::HttpApiKey& key) {
		FFL::CMutex::Autolock l(mApiListLock);
		std::list<ApiEntry*>::iterator it = mApiList.begin();
		for (; it != mApiList.end(); it++) {
			ApiEntry* entry = *it;
			if (entry->mKey.mPath== key.mPath && entry->mKey.mQuery == key.mQuery ) {
				mApiList.erase(it);
				return entry->mHandler;
			}
		}
		return NULL;
	}
	FFL::sp<HttpServer::Callback> HttpServerImpl::getRegisterApi(const HttpServer::HttpApiKey& key) {
		FFL::CMutex::Autolock l(mApiListLock);
		std::list<ApiEntry*>::iterator it = mApiList.begin();
		for (; it != mApiList.end(); it++) {
			ApiEntry* entry = *it;
			if (entry->mKey.mPath == key.mPath && entry->mKey.mQuery == key.mQuery) {
				return entry->mHandler;
			}
		}
		return NULL;
	}

	//
	//  内部保存的一条http连接上，的一下相关信息
	//
	class HttpContext {
	public:
		HttpContext(TcpClient* client);
		~HttpContext();

		TcpClient* mTcpClient;
		FFL::sp<HttpClient> mHttpClient;
	};
	HttpContext::HttpContext(TcpClient* client):mTcpClient(client){
		mHttpClient = new HttpClient (client);
	}
	HttpContext::~HttpContext() {				
		mTcpClient = NULL;
	}
	//
	//  一条http连接开始建立
	//
	bool HttpServerImpl::onClientCreate(TcpClient* client, int64_t* aliveTimeUs) {
		HttpContext* contex = new HttpContext(client);
		client->setUserdata(contex);
		return true;
	}
	void HttpServerImpl::onClientDestroy(TcpClient* client, TcpServer::Callback::FD_OPTMODE reason) {
		HttpContext* contex = (HttpContext*)client->getUserdata();
		client->setUserdata(NULL);

		if (reason == TcpServer::Callback::FD_REMOVE) {
			//
			//  需要httpclient管理了,
			//  当httpClient销毁的时候会自动销毁对应的client
			//
			contex->mHttpClient->setAutodelTcpClient(true);
		}

		FFL_SafeFree(contex);
	}
	//
	//  client上有数据可以读了
	//
	TcpServer::Callback::FD_OPTMODE HttpServerImpl::onClientReceived(TcpClient* client) {
		HttpContext* contex = (HttpContext*)client->getUserdata();
		FFL::sp<HttpRequest> request=contex->mHttpClient->readRequest();
		if (!request.isEmpty()) {
			if (!processHttpRequest(request)) {				
				//
				//  不再监控这个client的读事件了
				//
				return TcpServer::Callback::FD_REMOVE;
			}

		    //
			//  继续下一次的网络读事件产生
			//
			return TcpServer::Callback::FD_CONTINUE;
		}	
		//
		//  销毁这个client了
		//
		return TcpServer::Callback::FD_DESTROY;
	}
	//
	//  处理http请求
	//
	bool HttpServerImpl::processHttpRequest(FFL::sp<HttpRequest> request) {
		HttpUrl url;
		request->getUrl(url);

		HttpServer::HttpApiKey key;
		key.mPath = url.mPath;
		key.mQuery = url.mQuery;
		FFL::sp<HttpServer::Callback> apiHandler=getRegisterApi(key);
		if (!apiHandler.isEmpty()) {
			return apiHandler->onHttpQuery(request.get());
		} else {
			//
			//  404错误
			//
			FFL::sp<HttpResponse> response=request->makeResponse();
			response->setStatusCode(404);
			response->send();
			return false;
		}
		return true;
	}
	//
	//  调用。start，stop会触发onStart,onStop的执行
	//  onStart :表示准备开始了 ,可以做一些初始化工作
	//  onStop :表示准备停止了 ,可以做停止前的准备，想置一下信号让eventloop别卡住啊 
	//  在这些函数中，不要再调用自己的函数，例如：start,stop, isStarted等
	//
	FFL_INLINE bool HttpServerImpl::onStart() {
		mTcpServer->start(NULL);
		return true;
	}
	FFL_INLINE void HttpServerImpl::onStop() {
		mTcpServer->stop();
	}	
	//
	//   阻塞的线程中执行的eventloop,返回是否继续进行eventLoop
	//   waitTime:输出参数，下一次执行eventLoop等待的时长
	//   true  : 继续进行下一次的eventLoop
	//   false : 不需要继续执行eventloop
	//
	FFL_INLINE bool HttpServerImpl::eventLoop(int32_t* waitTime) {
		return mTcpServer->eventLoop(waitTime);
	}


	////////////////////////////////////////////////////////////////////////////////////////////////////////
	HttpServer::HttpServer(const char* ip, int32_t port) {
		mImpl = new HttpServerImpl(ip, port);
	}
	HttpServer::~HttpServer() {
		FFL_SafeFree(mImpl);
	}

	HttpServer::Callback::~Callback() {
	}

	//
	//  注册处理指定http ，请求的处理句柄
	//
	void HttpServer::registerApi(const HttpApiKey& key, FFL::sp<HttpServer::Callback> handler) {
		mImpl->registerApi(key, handler);
	}
	FFL::sp<HttpServer::Callback> HttpServer::unregisterApi(const HttpApiKey& key) {
		return mImpl->unregisterApi(key);
	}
	FFL::sp<HttpServer::Callback> HttpServer::getRegisterApi(const HttpApiKey& key) {
		return mImpl->getRegisterApi(key);
	}
	
	//
	//  调用。start，stop会触发onStart,onStop的执行
	//  onStart :表示准备开始了 ,可以做一些初始化工作
	//  onStop :表示准备停止了 ,可以做停止前的准备，想置一下信号让eventloop别卡住啊 
	//  在这些函数中，不要再调用自己的函数，例如：start,stop, isStarted等
	//
	bool HttpServer::onStart() {
		return mImpl->onStart();
	}
	void HttpServer::onStop() {
		mImpl->onStop();
	}
	
	//
	//   阻塞的线程中执行的eventloop,返回是否继续进行eventLoop
	//   waitTime:输出参数，下一次执行eventLoop等待的时长
	//   true  : 继续进行下一次的eventLoop
	//   false : 不需要继续执行eventloop
	//
	bool HttpServer::eventLoop(int32_t* waitTime) {
		return mImpl->eventLoop(waitTime);
	}
}

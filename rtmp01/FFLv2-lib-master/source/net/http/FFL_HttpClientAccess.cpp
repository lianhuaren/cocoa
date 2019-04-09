/*
*  This file is part of FFL project.
*
*  The MIT License (MIT)
*  Copyright (C) 2017-2018 zhufeifei All rights reserved.
*
*  FFL_HttpClientAccess.cpp
*  Created by zhufeifei(34008081@qq.com) on 2018/12/15
*  https://github.com/zhenfei2016/FFLv2-lib.git
*
*/

#include <net/http/FFL_HttpClientAccess.hpp>
#include <net/http/FFL_HttpRequest.hpp>
#include <net/http/FFL_HttpResponse.hpp>
#include <net/http/FFL_HttpParser.hpp>
#include <net/http/FFL_HttpClient.hpp>
#include <net/FFL_NetEventLoop.hpp>
#include <FFL_BlockList.hpp>
#include <list>
#include "internalLogConfig.h"

namespace FFL {

	class RequestContext : public RefBase {
	public:
		RequestContext();
		~RequestContext();

		int64_t mCreateTimeUs;
		FFL::sp<HttpRequest> mRequest;
		FFL::sp<HttpClientAccessManager::Callback> mCallback;

		TcpClient* mTcpClient;
		FFL::sp<HttpClient> mHttpClient;
	};
	RequestContext::RequestContext() :mCreateTimeUs(0), mTcpClient(NULL) {
	}
	RequestContext::~RequestContext() {
		if (mTcpClient) {
			mTcpClient->close();
			FFL_SafeFree(mTcpClient);
		}

		mHttpClient = NULL;
	}

	class HttpRequestThread;
	class HttpClientAccessManagerImpl : public NetEventLoop::Callback {
		friend class HttpRequestBuilder;
	public:
		HttpClientAccessManagerImpl();
		virtual ~HttpClientAccessManagerImpl();
	public:
		//
		//  启动，停止
		//
		bool start();
		void stop();
		//
		//  发送一个请求
		//
		bool post(FFL::sp<HttpRequest>& request, FFL::sp<HttpClientAccessManager::Callback> callback);
		/*  NetEventLoop::Callback  */
		//
		//  返回是否还继续读写
		//  priv:透传数据
		//
		virtual bool onNetEvent(NetFD fd, bool readable, bool writeable, bool exception, void* priv);
	protected:
		//
		//  请求队列
		//		
		FFL::BlockingList< FFL::sp<RequestContext> > mRequestQueue;

		//
		//  挂起，等待应答的请求列表
		//
		FFL::CMutex mPendingLock;
		std::list<  FFL::sp<RequestContext> > mPendingRequstList;
	protected:
		//
		//  处理这个请求
		//
		bool processRequest(FFL::sp<RequestContext> request);
	protected:
		//
		//  处理网络消息
		//
		NetEventLoop mEventLoop;

		//
		//  请求线程 
		//
		friend class HttpRequestThread;
		FFL::sp<HttpRequestThread> mHttpRequestThread;

	};
	//
	//  请求，连接发送线程
	//
	class HttpRequestThread : public  FFL::Thread {
	public:
		HttpRequestThread(HttpClientAccessManagerImpl* client) :mClient(client) {}
		bool threadLoop() {
			int32_t errorNo = 0;
			FFL::sp<RequestContext> entry = mClient->mRequestQueue.next(&errorNo);
			if (errorNo != 0) {
				//
				//  退出请求线程
				//
				return false;
			}

			if (!entry.isEmpty()) {
				mClient->processRequest(entry);
			}
			return true;
		}
		HttpClientAccessManagerImpl* mClient;
	};





	HttpClientAccessManagerImpl::HttpClientAccessManagerImpl() :
		mRequestQueue("httpRequestQueue"),
        mEventLoop(30 * 1000 * 1000)
		 {
	}
	HttpClientAccessManagerImpl::~HttpClientAccessManagerImpl() {

		FFL::CMutex::Autolock l(mPendingLock);
		mPendingRequstList.clear();
	}
	//
	//  启动，停止
	//
	bool HttpClientAccessManagerImpl::start() {
		if (!mHttpRequestThread.isEmpty()) {
			INTERNAL_FFL_LOG_DEBUG("HttpClientAccessManagerImpl: failed to start.");
			return false;
		}

		if (!mEventLoop.start(new ModuleThread("HttpClientAccessManagerImpl-eventloop"))) {
			INTERNAL_FFL_LOG_DEBUG("HttpClientAccessManagerImpl: failed to start eventloop .");
			return false;
		}

		mHttpRequestThread = new HttpRequestThread(this);
		mHttpRequestThread->run("HttpClientAccessManagerImpl-request");
		mRequestQueue.start();
		return true;
	}
	void HttpClientAccessManagerImpl::stop() {
		mRequestQueue.stop();
		mEventLoop.stop();
		if (!mHttpRequestThread.isEmpty()) {
			mHttpRequestThread->requestExitAndWait();
		}
		mRequestQueue.clear();
	}

	//
	//  发送一个请求
	//
	bool HttpClientAccessManagerImpl::post(FFL::sp<HttpRequest>& request, FFL::sp<HttpClientAccessManager::Callback> callback) {
		if (request.isEmpty()) {
			return false;
		}

		FFL::sp<RequestContext> entry = new RequestContext();
		entry->mRequest = request;
		entry->mCallback = callback;
		entry->mCreateTimeUs = FFL_getNowUs();
		if (!mRequestQueue.incoming(entry)) {
			return false;
		}
		return true;
	}
	/*  NetEventLoop::Callback  */
	//
	//  返回是否还继续读写
	//  priv:透传数据
	//
	bool HttpClientAccessManagerImpl::onNetEvent(NetFD fd, bool readable, bool writeable, bool exception, void* priv) {
		//
		//  找到请求
		//
		HttpClient* client = (HttpClient*)priv;
		FFL::sp<RequestContext> context;
		{
			FFL::CMutex::Autolock l(mPendingLock);
			for (std::list<  FFL::sp<RequestContext> >::iterator it = mPendingRequstList.begin();
				it != mPendingRequstList.end(); it++) {
				FFL::sp<RequestContext>& e = *it;
				if (e->mHttpClient.get() == client) {
					context = e;
					mPendingRequstList.erase(it);
					break;
				}
			}
		}

		if (context.isEmpty()) {
			return false;
		}

		//
		// 读应答
		//
		FFL::sp<HttpResponse> response = context->mHttpClient->readResponse();
		FFL::sp<HttpClientAccessManager::Callback> callback = context->mCallback;
		if (!callback.isEmpty()) {
			if (response.isEmpty()) {
				callback->onResponse(NULL, HttpClientAccessManager::Callback::ERROR_UNKONW);
			}
			else {
				callback->onResponse(response, HttpClientAccessManager::Callback::ERROR_SUC);
			}
		}		
		return false;
	}

	//
	//  处理这个请求
	//
	bool HttpClientAccessManagerImpl::processRequest(FFL::sp<RequestContext> entry) {
		FFL::HttpUrl url;
		entry->mRequest->getUrl(url);

		
		FFL::sp<HttpClient> httpClient=entry->mRequest->getHttpClient();
		if (httpClient.isEmpty()) {
			TcpClient* client = new TcpClient();
			if (client->connect(url.mHost.string(), url.mPort, *client) == FFL_OK) {				
				httpClient = new HttpClient(client);
				entry->mRequest->setHttpClient(httpClient);
				entry->mTcpClient = client;
				entry->mHttpClient = httpClient;
			}
		}

		if (!httpClient.isEmpty()) {
			NetFD fd = entry->mTcpClient->getFd();
			if (mEventLoop.addFd(fd, this, NULL, httpClient.get())) {
				if (entry->mRequest->send()) {
					//
					// 添加到等待应答的队列
					//
					FFL::CMutex::Autolock l(mPendingLock);
					mPendingRequstList.push_back(entry);
					return true;
				}
				mEventLoop.removeFd(fd);
			}
		}
		

		if (!entry->mCallback.isEmpty()) {
			entry->mCallback->onResponse(NULL, HttpClientAccessManager::Callback::ERROR_CONNECT);
		}		
		return false;
	}

	

	///////////////////////////////////////////////////////////////////////////////////////////////////

	HttpClientAccessManager::HttpClientAccessManager() {
		mImpl = new HttpClientAccessManagerImpl();
	}
	HttpClientAccessManager::~HttpClientAccessManager(){
		FFL_SafeFree(mImpl);
	}
   
	//
	//  启动，停止
	//
	bool HttpClientAccessManager::start() {
		return mImpl->start();
	}
	void HttpClientAccessManager::stop() {
		return mImpl->stop();
	}
	//
	//  发送一个请求
	//
	bool HttpClientAccessManager::post(FFL::sp<HttpRequest>& request, FFL::sp<Callback> callback) {
		return mImpl->post(request,callback);
	}
   
}

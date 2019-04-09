/*
*  This file is part of FFL project.
*
*  The MIT License (MIT)
*  Copyright (C) 2017-2018 zhufeifei All rights reserved.
*
*  FFL_HttpServer.hpp
*  Created by zhufeifei(34008081@qq.com) on 2018/11/08
*  https://github.com/zhenfei2016/FFLv2-lib.git
*
*  http服务器
*/

#ifndef _FFL_HTTP_SERVER_HPP_
#define _FFL_HTTP_SERVER_HPP_

#include <net/FFL_TcpServer.hpp>
#include <FFL_ByteStream.hpp>
#include <net/http/FFL_HttpRequest.hpp>
#include <net/http/FFL_HttpResponse.hpp>
#include <net/http/FFL_HttpFile.hpp>

namespace FFL {        
	class HttpServerImpl;
    class FFLIB_API_IMPORT_EXPORT HttpServer : public Module{
	public:
		class FFLIB_API_IMPORT_EXPORT Callback: public RefBase {
		public:
			virtual ~Callback();
			//
			//  返回false则强制关闭这个连接
			//
			virtual bool onHttpQuery(HttpRequest* request) = 0;
		};
    public:
        HttpServer(const char* ip, int32_t port);
        virtual ~HttpServer();        
    public:
        class HttpApiKey{
        public:
            String mPath;
            String mQuery;
        };
        //
        //  注册处理指定http ，请求的处理句柄
        //
        void registerApi(const HttpApiKey& key, FFL::sp<HttpServer::Callback> handler);
        FFL::sp<HttpServer::Callback> unregisterApi(const HttpApiKey& key);
        FFL::sp<HttpServer::Callback> getRegisterApi(const HttpApiKey& key);
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
		HttpServerImpl* mImpl;
	};
}

#endif
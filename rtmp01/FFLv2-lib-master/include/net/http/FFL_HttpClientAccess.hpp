/*
*  This file is part of FFL project.
*
*  The MIT License (MIT)
*  Copyright (C) 2017-2018 zhufeifei All rights reserved.
*
*  FFL_HttpClientManager.hpp
*  Created by zhufeifei(34008081@qq.com) on 2018/12/15
*  https://github.com/zhenfei2016/FFLv2-lib.git
*  http请求 sender  ,用于请求HttpRequest
*
*/

#ifndef _FFL_HTTP_CLIENT_ACCESS_MANGER_HPP_
#define _FFL_HTTP_CLIENT_ACCESS_MANGER_HPP_

#include <net/FFL_TcpClient.hpp>
#include <net/FFL_NetStream.hpp>
#include <FFL_RefBase.hpp>

namespace FFL {   
   class HttpRequest;
   class HttpResponse;     
   class HttpClientAccessManagerImpl;

   class FFLIB_API_IMPORT_EXPORT HttpClientAccessManager {
	   friend class HttpRequestBuilder;
   public:
	   class Callback : public RefBase{
	   public:
		   enum {
			   //
			   //  没有错误
			   //
			   ERROR_SUC=0,
			   //
			   //  连接服务器失败
			   //
			   ERROR_CONNECT=-1,

			   ERROR_UNKONW =-2,
		   };
	   public:
		   //
		   //  网络应答
		   //  errorNo :错误码
		   //
		   virtual void onResponse(FFL::sp<HttpResponse> response,int32_t errorNo) = 0;
	   };
    public:
		HttpClientAccessManager();
		~HttpClientAccessManager();
   public:
		//
		//  启动，停止
	    //
		bool start();
		void stop();
		//
		//  发送一个请求
		//
		bool post(FFL::sp<HttpRequest>& request,FFL::sp<Callback> callback);
   private:
	   HttpClientAccessManagerImpl* mImpl;
   };  
}
#endif

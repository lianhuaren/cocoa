/*
*  This file is part of FFL project.
*
*  The MIT License (MIT)
*  Copyright (C) 2017-2018 zhufeifei All rights reserved.
*
*  FFL_NetTcpServer.cpp
*  Created by zhufeifei(34008081@qq.com) on 2018/07/14
*  https://github.com/zhenfei2016/FFLv2-lib.git
*  网路服务器基类
*
*/

#ifndef _FFL_HTTP_CLIENT_HPP_
#define _FFL_HTTP_CLIENT_HPP_

#include <net/FFL_TcpClient.hpp>
#include <net/FFL_NetStream.hpp>
#include <ref/FFL_RefBase.hpp>

namespace FFL {   
   class HttpRequest;
   class HttpResponse;  
  
   class HttpClient : public RefBase{
   protected:
		HttpClient(TcpClient* client);
   public:
		~HttpClient();
		//
		//  读取一个请求
		//
		FFL::sp<HttpRequest> readRequest();
		//
		//  发送一个请求
		//
		bool writeRequest(FFL::sp<HttpRequest>& request);
		//
		//  读取一个应答
		//
		FFL::sp<HttpResponse> readResponse();
		//
		//  发送一个应答
		//
		bool writeResponse(FFL::sp<HttpResponse>& request);
	protected:
		friend class HttpResponse;
		friend class HttpRequest;
	   //
	   //  读取内容，
	   //
	   bool read(char* data, int32_t requstSize, size_t* readed);
	   //
	   //  写内容
	   //
	   bool write(const char* data, int32_t requstSize, size_t* writed);
	   //
	   //  关闭这个连接
	   //
	   void close();
   protected:
	   TcpClient* mClient;
	   NetStreamReader mData;
   };  
}
#endif

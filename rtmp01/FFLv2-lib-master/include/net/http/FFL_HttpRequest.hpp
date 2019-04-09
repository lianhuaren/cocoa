/*
*  This file is part of FFL project.
*
*  The MIT License (MIT)
*  Copyright (C) 2017-2018 zhufeifei All rights reserved.
*
*  FFL_HttpRequest.hpp   
*  Created by zhufeifei(34008081@qq.com) on 2018/07/15
*  https://github.com/zhenfei2016/FFL-v2.git
*
*  http请求
*/
#ifndef _FFL_HTTP_REQUEST_HPP_
#define _FFL_HTTP_REQUEST_HPP_

#include <FFL_Core.h>
#include <FFL_Ref.hpp>
#include <net/http/FFL_HttpTransportBase.hpp>
#include <net/FFL_NetStream.hpp>

namespace FFL {
	class HttpResponse;

	enum  HTTP_Method {
		GET = 0,
		POST = 1,
	};

	class FFLIB_API_IMPORT_EXPORT HttpRequest : public HttpTransportBase {
	public:
		friend class HttpClient;
		friend class HttpParserImpl;
		
		HttpRequest();
		HttpRequest(FFL::sp<HttpClient> client);
	public:
		virtual ~HttpRequest();	
		FFL::sp<HttpResponse> makeResponse();
	public:
		//
		//  获取设置method
		//
		void setMethod(HTTP_Method med);
		HTTP_Method getMethod();
		//
		//  请求内容
		//  header :头内容		
		//  content:内容
		//
		virtual bool writeHeader();
	protected:
		HTTP_Method mMethod;
	};
}

#endif
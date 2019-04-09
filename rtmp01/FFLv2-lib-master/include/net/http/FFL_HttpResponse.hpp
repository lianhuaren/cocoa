/*
*  This file is part of FFL project.
*
*  The MIT License (MIT)
*  Copyright (C) 2017-2018 zhufeifei All rights reserved.
*
*  FFL_HttpResponse.hpp   
*  Created by zhufeifei(34008081@qq.com) on 2018/07/14
*  https://github.com/zhenfei2016/FFL-v2.git
*
*  http应答
*/
#ifndef _FFL_HTTP_RESPONSE_HTTP_HPP_
#define _FFL_HTTP_RESPONSE_HTTP_HPP_


#include <FFL_Core.h>
#include <FFL_Ref.hpp>
#include <net/http/FFL_HttpHeader.hpp>
#include <net/http/FFL_HttpTransportBase.hpp>

namespace FFL {
	class HttpClient;
	class ByteBuffer;	
	class FFLIB_API_IMPORT_EXPORT HttpResponse : public HttpTransportBase{
	public:
	    friend class HttpClient;
		friend class HttpRequest;
		friend class HttpParserImpl;

		//
		//  应答必须附加到一个client上的
		//
		HttpResponse(FFL::sp<HttpClient> client);	
		virtual ~HttpResponse();			
	public:
		//
		//  获取，设置 状态码
		//				
		int32_t getStatusCode();
		void setStatusCode(int32_t code);		
	protected:
		//
		//  请求内容
		//  header :头内容		
		//  content:内容
		//
		bool writeHeader();			
	protected:
		//
		//  状态码
		//
		int32_t mStatusCode;		
	};		
}

#endif
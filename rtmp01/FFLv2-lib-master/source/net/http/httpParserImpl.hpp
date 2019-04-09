/*
*  This file is part of FFL project.
*
*  The MIT License (MIT)
*  Copyright (C) 2017-2018 zhufeifei All rights reserved.
*
*  httpParserImpl.hpp
*  Created by zhufeifei(34008081@qq.com) on 2018/07/15
*  https://github.com/zhenfei2016/FFL-v2.git
*
*  http分析实现类，封装了开源库http-parser-2.1
*/

#ifndef _FFL_HTTPPARSERIMPL_HPP_
#define _FFL_HTTPPARSERIMPL_HPP_

#include <FFL_Core.h>
#include <net/http/FFL_HttpRequest.hpp>
#include <net/http/FFL_HttpResponse.hpp>
#include "http-parser-2.1/http_parser.h"

namespace FFL {
	class HttpParserImpl {
	public:
		HttpParserImpl();
		~HttpParserImpl();

		void init(enum http_parser_type type);

		status_t parseRequest(NetStreamReader* stream,HttpRequest* request);
		status_t parseResponse(NetStreamReader* stream,HttpResponse* reponse);

	public:
		
		static int onMessageBegin(http_parser* parser);
		static int onUrl(http_parser* parser, const char* at, size_t length);
		static int onHeaderField(http_parser* parser, const char* at, size_t length);
		static int onHeaderValue(http_parser* parser, const char* at, size_t length);
		static int onHeadersComplete(http_parser* parser);
		static int onBody(http_parser* parser, const char* at, size_t length);
		static int onMessageComplete(http_parser* parser);

	public:
		http_parser_settings mSettings;
		http_parser mParser;
		//
		// 请求的url
		//
		String mUrl;
		//
		// 临时变量，parser用到
		//
		String mTmpFieldName;
		String mTmpFieldValue;
		//
		//  parser的当前状态
		//
		enum HTTPState {
			PARSER_BEGIN,
			PARSE_HeaderComplete,
			PARSE_COMPLETE,

		};
		HTTPState mState;	
		HttpHeader* mHeader;
	};
}

#endif
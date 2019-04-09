/*
*  This file is part of FFL project.
*
*  The MIT License (MIT)
*  Copyright (C) 2017-2018 zhufeifei All rights reserved.
*
*  FFL_HttpParser.hpp   
*  Created by zhufeifei(34008081@qq.com) on 2018/07/19 
*  https://github.com/zhenfei2016/FFL-v2.git
*
*  分析http请求，分离header /body等
*/
#ifndef _HTTP_NET_PARSER_HTTP_HPP_
#define _HTTP_NET_PARSER_HTTP_HPP_

#include <net/http/FFL_HttpRequest.hpp>
#include <net/http/FFL_HttpResponse.hpp>


namespace FFL {	
	class HttpParserImpl;	
	class FFLIB_API_IMPORT_EXPORT HttpParser
	{	
	public:
		HttpParser();
		virtual ~HttpParser();
	public:		
		//
		//  stream :读取数据，分析请求头
		//  parser http请求，成功返回FFL_OK
		//
		status_t parseRequest(NetStreamReader* stream, HttpRequest* request);
		//
		//  parser http应答，成功返回FFL_OK
		//
		status_t parseResponse(NetStreamReader* stream, HttpResponse* response);
	private:	
		//
		//  获取头信息
		//
		HttpHeader* getHeader() const {
			return mHeader;
		}
	private:
		HttpParserImpl* mImpl;
		HttpHeader* mHeader;
	};
}

#endif
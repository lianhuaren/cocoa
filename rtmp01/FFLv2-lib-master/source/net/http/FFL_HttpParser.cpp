/*
*  This file is part of FFL project.
*
*  The MIT License (MIT)
*  Copyright (C) 2017-2018 zhufeifei All rights reserved.
*
*  FFL_HttpParser.hpp
*  Created by zhufeifei(34008081@qq.com) on 2018/07/19
*  https://github.com/zhenfei2016/FFLv2-lib.git
*
*  分析http请求，分离header /body等
*/

#include <FFL_String.hpp>
#include <FFL_Dictionary.hpp>
#include <FFL_ByteStream.hpp>
#include <net/http/FFL_HttpParser.hpp>
#include <net/FFL_NetStream.hpp>
#include "httpParserImpl.hpp"

namespace FFL{
	HttpParser::HttpParser(){
		mImpl = new HttpParserImpl();
		mImpl->init(HTTP_BOTH);
	}
	HttpParser::~HttpParser(){
		FFL_SafeFree(mImpl);
		mImpl = NULL;
	}
	//
	//  parser http请求，成功返回FFL_OK
	//
	status_t HttpParser::parseRequest(NetStreamReader* stream, HttpRequest* request) {
		return mImpl->parseRequest(stream,request);
	}
	//
	//  parser http应答，成功返回FFL_OK
	//
	status_t HttpParser::parseResponse(NetStreamReader* stream, HttpResponse* response) {
		return mImpl->parseResponse(stream,response);
	}

}
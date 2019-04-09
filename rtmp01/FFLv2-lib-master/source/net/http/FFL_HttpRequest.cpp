/*
*  This file is part of FFL project.
*
*  The MIT License (MIT)
*  Copyright (C) 2017-2018 zhufeifei All rights reserved.
*
*  FFL_HttpRequest.cpp
*  Created by zhufeifei(34008081@qq.com) on 2018/07/15
*  https://github.com/zhenfei2016/FFLv2-lib.git
*
*  http请求
*/
#include <net/http/FFL_HttpRequest.hpp>
#include <net/http/FFL_HttpResponse.hpp>
#include <net/http/FFL_HttpClient.hpp>

namespace FFL {	
	/////////////////////////////////////////////////////////////////////////
	HttpRequest::HttpRequest():HttpTransportBase(NULL) {
		mMethod = POST;
	}
	HttpRequest::HttpRequest(FFL::sp<HttpClient> client):
		HttpTransportBase(client){
		mMethod = POST;
	}
	HttpRequest::~HttpRequest() {
	}
	FFL::sp<HttpResponse> HttpRequest::makeResponse() {
		return new HttpResponse(mClient);
	}
	//
	//  获取设置method
	//
	void HttpRequest::setMethod(HTTP_Method med) {
		mMethod = med;
	}
	HTTP_Method HttpRequest::getMethod() {
		return mMethod;
	}
	//
	//  请求内容
	//  header :头内容		
	//  content:内容
	//
	bool HttpRequest::writeHeader() {
		if (mClient.isEmpty()) {
			return false;
		}

		//
		//  发送的内容长度
		//
		int32_t contentLength = 0;
		if (!mContent.isEmpty()) {
			contentLength=mContent->getSize();
		}else {
			contentLength=mHeader.getContentLength();
		}
		if (contentLength < 0) {
			contentLength = 0;
		}

		String headerInfo;
		{
			static const char* format =
				"%s %s HTTP/1.1\r\n";
				//"%s: %s\r\n"
				//"%s: %d\r\n";

			String type;
			mHeader.getContentType(type);

			String path;
			if (mUrl.mPath.isEmpty()){
				path = "/";
			}
			else {
				path = mUrl.mPath + "?" + mUrl.mQuery;
			}

			headerInfo = String::format(format, 
				getMethod()==POST? "POST" : "GET",
				path.string(),
				HTTP_KEY_CONTENTYPE, type.string(),
				HTTP_KEY_CONTENTLEN, contentLength);

			headerInfo = String::format(format,
				getMethod() == POST ? "POST" : "GET",
				path.string());
		}

		int32_t buffCount = 20;
		FFL::Dictionary::Pair pairs[20];
		mHeader.getAll(pairs, &buffCount);

		for (int32_t i = 0; i < buffCount; i++) {
			if (strcmp(HTTP_KEY_CONTENTYPE, pairs[i].key.string()) == 0) {
				continue;
			}
			else if (strcmp(HTTP_KEY_CONTENTLEN, pairs[i].key.string()) == 0) {
				continue;
			}
			headerInfo += pairs[i].key + ":" + pairs[i].value + "\r\n";
		}


		headerInfo += "\r\n";
		return mClient->write(headerInfo.string(), headerInfo.size(), 0);
	}
}


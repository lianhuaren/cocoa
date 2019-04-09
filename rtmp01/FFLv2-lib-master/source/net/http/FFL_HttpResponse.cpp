/*
*  This file is part of FFL project.
*
*  The MIT License (MIT)
*  Copyright (C) 2017-2018 zhufeifei All rights reserved.
*
*  FFL_HttpResponse.cpp
*  Created by zhufeifei(34008081@qq.com) on 2018/07/14
*  https://github.com/zhenfei2016/FFLv2-lib.git
*
*  http应答
*/

#include <net/http/FFL_HttpResponse.hpp>
#include <net/http/FFL_HttpClient.hpp>

namespace FFL {
	//const char* JSON_HEADER = "HTTP/1.1 200 OK \r\n"
	//	"Content-Type: application/json;charset=utf-8\r\n";
	HttpResponse::HttpResponse(FFL::sp<HttpClient> client):
		   HttpTransportBase(client){
		mClient=client;
		mStatusCode = 200;		
	}
	HttpResponse::~HttpResponse(){
	}	
	//
	//  状态码
	//				
	int32_t HttpResponse::getStatusCode() {
		return mStatusCode;
	}
	void HttpResponse::setStatusCode(int32_t code) {
		mStatusCode = code;
	}
	///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////	
	//
	//  请求内容
	//  header :头内容		
	//  content:内容
	//
	bool HttpResponse::writeHeader() {	
		//
		//  发送的内容长度
		//
		int32_t contentLength = 0;
		if (!mContent.isEmpty()) {
			contentLength = mContent->getSize();
		}
		else {
			contentLength = mHeader.getContentLength();
		}
		if (contentLength < 0) {
			contentLength = 0;
		}

		String headerInfo;
		{			
			static const char* kFormat= "HTTP/1.1 %d OK \r\n"
				"%s: %s\r\n"
				"%s: %d\r\n";
			String type;
			mHeader.getContentType(type);
			headerInfo.appendFormat(kFormat,mStatusCode,
				HTTP_KEY_CONTENTYPE,type.string(),
				HTTP_KEY_CONTENTLEN,contentLength);
		}
				
		int32_t buffCount = 20;
		FFL::Dictionary::Pair pairs[20];
		mHeader.getAll(pairs,&buffCount);

		for (int32_t i = 0; i < buffCount;i++) {
			if (strcmp(HTTP_KEY_CONTENTYPE, pairs[i].key.string()) == 0) {
				continue;
			}
			else if (strcmp(HTTP_KEY_CONTENTLEN, pairs[i].key.string()) == 0) {
				continue;
			}
			headerInfo += pairs[i].key + ":" + pairs[i].value + "\r\n";
		}
		
		headerInfo += "\r\n";		
		return mClient->write(headerInfo.string(), headerInfo.size(),0);
	}
	
}

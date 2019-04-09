/*
*  This file is part of FFL project.
*
*  The MIT License (MIT)
*  Copyright (C) 2017-2018 zhufeifei All rights reserved.
*
*  FFL_HttpHeader.cpp
*  Created by zhufeifei(34008081@qq.com) on 2018/07/14
*  https://github.com/zhenfei2016/FFLv2-lib.git
*  http头信息
*
*/

#include <net/http/FFL_HttpHeader.hpp>


namespace FFL {	
	
	HttpHeader::HttpHeader(){
		mContentType = "text/plain;charset=utf-8";
		mContentLength = 0;
	}
	HttpHeader::~HttpHeader(){}

	//
	// 获取content类型
	//
	void HttpHeader::getContentType(String& type)  {
		type = mContentType;

		if (type.size() == 0) { 
			this->getKey(HTTP_KEY_CONTENTYPE, mContentType);
		}
	}
	void HttpHeader::setContentType(const String& type) {
		mContentType = type;
	}
	//
	// 获取content的长度
	//
	int32_t HttpHeader::getContentLength()  {

		if (mContentLength == 0) {
			getKeyInt32(HTTP_KEY_CONTENTLEN, mContentLength, 0);
		}
		return mContentLength;
	}
	void HttpHeader::setContentLength(int32_t len) {
		mContentLength = len;
	}


	HttpContent::HttpContent(){}
	HttpContent::~HttpContent(){}

	
}

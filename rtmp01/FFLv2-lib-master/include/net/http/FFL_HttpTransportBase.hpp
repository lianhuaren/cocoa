/*
*  This file is part of FFL project.
*
*  The MIT License (MIT)
*  Copyright (C) 2017-2018 zhufeifei All rights reserved.
*
*  FFL_HttpTransportBase.hpp   
*  Created by zhufeifei(34008081@qq.com) on 2019/02/19
*  https://github.com/zhenfei2016/FFL-v2.git
*
*  http请求,应答的基础通讯接口实现
*/
#ifndef _FFL_HTTP_TRANSPORT_BASE_HPP_
#define _FFL_HTTP_TRANSPORT_BASE_HPP_

#include <FFL_Core.h>
#include <FFL_Ref.hpp>
#include <net/http/FFL_HttpHeader.hpp>
#include <net/http/FFL_HttpUrl.hpp>
#include <net/http/FFL_HttpClient.hpp>

namespace FFL {

	class FFLIB_API_IMPORT_EXPORT HttpTransportBase : public RefBase {
	public:		
		HttpTransportBase(FFL::sp<HttpClient> client);
		virtual ~HttpTransportBase();		
	public:	
		//
		//  指定client连接，发送接收的数据是在这一条连接上执行的 
		//
		void setHttpClient(FFL::sp<HttpClient> client);
		FFL::sp<HttpClient> getHttpClient();
		//
		//  请求参数相关
		//				
		void getUrl(HttpUrl& url);
		void setUrl(HttpUrl& url);		
		//
		//  头信息
		//
		void getHeader(HttpHeader& header);
		void setHeader(HttpHeader& header);
		//
		//  设置内容，用于发送端,如果数据量小可以直接设置data,size
		//  数据量大的情况下需要设置ContentStream指针
		//
		void setContent(const uint8_t* data, uint32_t size);
		void setContent(FFL::sp<HttpContent> content);
		//
		//  开始发送请求
		//  执行了写头，写内容操作 writeHeader();writeContent();
		//  writeContent会依次的从HttpContent中读取需要发送的内容的
		//
		bool send();
		//
		//  返回读取的内容，用于接收端的,返回的HttpContent需要载保证
		//  HttpTransportBase有效的情况下使用
		//
		FFL::sp<HttpContent> readContent();
		//
		//  结束应答,关闭这个连接
		//
		void finish();
	protected:
		//
		//  请求内容
		//  header :头内容		
		//  content:内容
		//
		virtual bool writeHeader();
		virtual bool writeContent();	
	protected:
		//
		//  请求头
		//
		HttpHeader mHeader;	
		// parse uri to schema/server:port/path?query
		HttpUrl mUrl;	
		//
		//  发送的内容
		//
		FFL::sp<HttpContent> mContent;
		//
		//  那个连接上的请求
		//
		FFL::sp<HttpClient> mClient;
	};
}

#endif
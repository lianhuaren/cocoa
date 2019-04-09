/*
*  This file is part of FFL project.
*
*  The MIT License (MIT)
*  Copyright (C) 2017-2018 zhufeifei All rights reserved.
*
*  FFL_HttpTransportBase.cpp
*  Created by zhufeifei(34008081@qq.com) on 2019/02/19
*  https://github.com/zhenfei2016/FFL-v2.git
*
*  http请求,应答的基础通讯接口实现
*/
#include <net/http/FFL_HttpRequest.hpp>
#include <net/http/FFL_HttpResponse.hpp>
#include <net/http/FFL_HttpClient.hpp>
#include <FFL_ByteBuffer.hpp>
#include <FFL_ByteStream.hpp>
#include "httpSimpleContent.hpp"

namespace FFL {		
	class HttpClientContent : public HttpContent {
	public:
		HttpClientContent(HttpClient* clent, int32_t size);
		~HttpClientContent();
	public:
		//
		//  获取内容大小
		//
		int32_t getSize();
		//
		//  获取内容
		//
		int32_t read(uint8_t* data, int32_t requestSize, bool* suc);
	protected:
		HttpClient* mClient;
		int32_t mSize;
	};

	HttpClientContent::HttpClientContent(HttpClient* client,int32_t size):
		mClient(client),
		mSize (size){
		
	}
	HttpClientContent::~HttpClientContent(){}

	//
	//  获取内容大小
	//
	int32_t HttpClientContent::getSize() {
		return mSize;
	}
	//
	//  获取内容
	//
	int32_t HttpClientContent::read(uint8_t* data, int32_t requestSize, bool* suc) {
		if (suc) {
			*suc = false;
		}

		if (!mClient) {			
			return 0;
		}
		size_t readed = 0;			
		if (mClient->read((char*)data, requestSize, &readed)) {
			if (suc) *suc = true;
			return readed;
		}

		return readed;
	}

	///////////////////////////////////////////////////////////////////////
	///////////////////////////////////////////////////////////////////////
	HttpTransportBase::HttpTransportBase(FFL::sp<HttpClient> client):
		   mClient(client){		
	}
	HttpTransportBase::~HttpTransportBase() {
	}
	void HttpTransportBase::setHttpClient(FFL::sp<HttpClient> client) {
		mClient = client;
	}
	FFL::sp<HttpClient> HttpTransportBase::getHttpClient() {
		return mClient;
	}
	//
	//  请求参数相关
	//				
	void HttpTransportBase::getUrl(HttpUrl& url) {
		url = mUrl;
	}
	void HttpTransportBase::setUrl(HttpUrl& url) {
		mUrl = url;
	}
	void HttpTransportBase::getHeader(HttpHeader& header) {
		header = mHeader;
	}
	void HttpTransportBase::setHeader(HttpHeader& header) {
		mHeader = header;
	}
	//
	//  设置内容，用于发送端
	//
	void HttpTransportBase::setContent(const uint8_t* data, uint32_t size) {
		if (data && size > 0) {
			mContent = new HttpSimpleContent(data,(int32_t) size);
		}
		else {
			mContent = NULL;
		}
	}
	void HttpTransportBase::setContent(FFL::sp<HttpContent> content) {
		mContent = content;
	}

	//
	//  开始发送请求，
	//
	bool HttpTransportBase::send() {
		if (writeHeader()) {
			return writeContent();
		}
		return false;		
	}	
	//
	//  返回读取的内容，用于接收端的
	//
	FFL::sp<HttpContent> HttpTransportBase::readContent() {
		int32_t contentLength = mHeader.getContentLength();
		return new HttpClientContent(mClient.get(), contentLength);
	}
	//
	//  结束应答,关闭这个连接
	//
	void HttpTransportBase::finish() {
		if (!mClient.isEmpty()) {
			mClient->close();
			mClient = NULL;
		}

		mContent = NULL;
	}
	//
	//  请求内容
	//  header :头内容		
	//  content:内容
	//
	bool HttpTransportBase::writeHeader() {
		if (mClient.isEmpty()) {
			return false;
		}

		//
		//  发送的内容长度，优先使用内容类指向的大小
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
				"POST %s HTTP/1.1\r\n"
				"%s: %s\r\n"
				"%s: %d\r\n";

			String type;
			mHeader.getContentType(type);

			headerInfo = String::format(format, 
				(mUrl.mPath + "?" + mUrl.mQuery).string(),
				HTTP_KEY_CONTENTYPE, type.string(),
				HTTP_KEY_CONTENTLEN, contentLength);
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
	//
	//  写内容数据
	//
	bool HttpTransportBase::writeContent() {
		if (mClient.isEmpty()) {
			return false;
		}

		if (mContent.isEmpty()) {
			return true;
		}

		const static int32_t BLOCK_SIZE = 4096;
		uint8_t buf[BLOCK_SIZE] = {};		
		int32_t bufSize = BLOCK_SIZE;
		int32_t len=mContent->getSize();
		while(len>0){
			bufSize = len<BLOCK_SIZE ? len: BLOCK_SIZE;
			int32_t readed = mContent->read(buf, bufSize, NULL);
			if(readed<=0){
				break;
			}

			if (!mClient->write((const char*)buf, readed, 0)) {
				break;
			}

			len -= readed;
			FFL_sleep(5);
		}
		
		return true;
	}		
}


/*
*  This file is part of FFL project.
*
*  The MIT License (MIT)
*  Copyright (C) 2017-2018 zhufeifei All rights reserved.
*
*  httpParserImpl.cpp   
*  Created by zhufeifei(34008081@qq.com) on 2018/07/15
*  https://github.com/zhenfei2016/FFLv2-lib.git
*
*  http分析实现类，封装了开源库http-parser-2.1
*/
#include "httpParserImpl.hpp"
#include <net/http/FFL_HttpHeader.hpp>
#include <net/FFL_NetStream.hpp>
#include "internalLogConfig.h"

namespace FFL{
	HttpParserImpl::HttpParserImpl() {

	}
	HttpParserImpl::~HttpParserImpl() {

	}

	void HttpParserImpl::init(enum http_parser_type type) {
		memset(&mSettings, 0, sizeof(mSettings));
		mSettings.on_message_begin = onMessageBegin;
		mSettings.on_url = onUrl;
		mSettings.on_header_field = onHeaderField;
		mSettings.on_header_value = onHeaderValue;
		mSettings.on_headers_complete = onHeadersComplete;
		mSettings.on_body = onBody;
		mSettings.on_message_complete = onMessageComplete;
		http_parser_init(&mParser, type);
		mParser.data = (void*)this;
	}

	static bool haveKeepAlive(const http_parser *parser){
		if (parser->http_major > 0 && parser->http_minor > 0) {
			/* HTTP/1.1 */
			if (parser->flags & F_CONNECTION_CLOSE) {
				return 0;
			}
		}
		else {
			/* HTTP/1.0 or earlier */
			if (!(parser->flags & F_CONNECTION_KEEP_ALIVE)) {
				return 0;
			}
		}

		return 1;
	}

	//0d0a0d0a
	const static uint8_t KEYSET_CR = 13;
	const static uint8_t KEYSET_LF = 10;
	status_t HttpParserImpl::parseRequest(NetStreamReader* stream,HttpRequest* request) {		
		status_t ret = FFL_OK;
		mHeader = &(request->mHeader);

		while (true) {
			const uint8_t* data= stream->getData() + stream->getPosition();
			uint32_t size =stream->getSize();
			size_t nParsed = 0;

			//
			// 找到http头结束位置，开始分析到头结束位置
			//
			if (size > 4) {
				for (int32_t i = 0; i <= (int32_t)size - 4; i++) {
					const uint8_t* src = data + i;
					if (src[0] == KEYSET_CR &&src[1] == KEYSET_LF &&src[2] == KEYSET_CR &&src[3] == KEYSET_LF) {
						nParsed = http_parser_execute(&mParser, &mSettings, (const char*)data, i + 4);
						break;
					}
				}
			}

			if (nParsed>0 && (PARSE_HeaderComplete == mState ||
				PARSE_COMPLETE == mState) ) {
				stream->skip(nParsed);
				break;
			}

			//
			//  查看数据是否要溢出了，如果要溢出则表示数据不对
			//
			// stream->getPosition() + stream->getSize()

			//
			//  数据不够，拉取数据
			//
			if (nParsed == 0 ) {
				if ((ret=stream->pull(-1)) != FFL_OK) {
					return ret;
				}
			}
		}

		//
		//  填充请求信息
		//
		if (request) {	
			String url;
			String host;
			if (mHeader->getKey("Host", host)) {
				url = "http://";
				url +=host + mUrl;
			}
			else {
				url = mUrl;
			}
			request->mUrl.parse(url);		

			String contentType;
			mHeader->getKey(HTTP_KEY_CONTENTYPE, contentType);
			mHeader->setContentType(contentType);

			int32_t len = 0;
			mHeader->getKeyInt32(HTTP_KEY_CONTENTLEN, len, 0);
			mHeader->setContentLength(len);
		}

		return FFL_OK;
	}
	status_t HttpParserImpl::parseResponse(NetStreamReader* stream,HttpResponse* response) {
		status_t ret = FFL_OK;
		mHeader = &(response->mHeader);

		while (true) {
			const uint8_t* data = stream->getData() + stream->getPosition();
			uint32_t size = stream->getSize();
			size_t nParsed = 0;

			//
			// 找到http头结束位置，开始分析到头结束位置
			//
			if (size > 4) {
				for (int32_t i = 0; i <= (int32_t)size - 4; i++) {
					const uint8_t* src = data + i;
					if (src[0] == KEYSET_CR &&src[1] == KEYSET_LF &&src[2] == KEYSET_CR &&src[3] == KEYSET_LF) {
						nParsed = http_parser_execute(&mParser, &mSettings, (const char*)data, i + 4);
						break;
					}
				}
			}

			if (nParsed > 0 && (PARSE_HeaderComplete == mState ||
				PARSE_COMPLETE == mState)) {
				stream->skip(nParsed);
				break;
			}
						
			//
			//  数据不够，拉取数据
			//
			if (nParsed == 0) {
				if ((ret = stream->pull(-1)) != FFL_OK) {
					return ret;
				}
			}
		}

		//
		//  填充请求信息
		//
		if (response) {
			String url;
			String host;
			if (mHeader->getKey("Host", host)) {
				url = "http://";
				url += host + mUrl;
			}
			else {
				url = mUrl;
			}
			//response->mUrl.parse(url.c_str());
		}

		mHeader = NULL;

		return FFL_OK;
	}

	static HttpParserImpl* getThis(http_parser* parser) {
		return (HttpParserImpl*)parser->data;
	}
	int HttpParserImpl::onMessageBegin(http_parser* parser) {
		HttpParserImpl* pThis = getThis(parser);
		pThis->mState = PARSER_BEGIN;
		return 0;
	}
	int HttpParserImpl::onUrl(http_parser* parser, const char* at, size_t length) {
		HttpParserImpl* pThis = getThis(parser);
		if (length > 0) {
			pThis->mUrl.append(at, (int)length);
			//FFL_LOG_DEBUG("HttpParser: url=%s", pThis->mUrl.c_str());
		}
		return 0;
	}
	int HttpParserImpl::onHeaderField(http_parser* parser, const char* at, size_t length) {
		HttpParserImpl* pThis = getThis(parser);
		if (length > 0) {
			if (pThis->mTmpFieldName.size()) {
				pThis->mTmpFieldName = "";
			}
			pThis->mTmpFieldName.append(at, (int)length);
			//FFL_LOG_DEBUG("HttpParser: FieldName=%s", pThis->mTmpFieldName.c_str());
		}
		return 0;
	}
	int HttpParserImpl::onHeaderValue(http_parser* parser, const char* at, size_t length) {
		HttpParserImpl* pThis = getThis(parser);
		if (length > 0) {
			if (pThis->mTmpFieldValue.size()) {
				pThis->mTmpFieldValue = "";
			}
			pThis->mTmpFieldValue.append(at, (int)length);
			//FFL_LOG_DEBUG("HttpParser: FieldValue=%s", pThis->mTmpFieldValue.c_str());
		}

		if (pThis->mTmpFieldName.size() && pThis->mTmpFieldValue.size()) {
			pThis->mHeader->setKey(pThis->mTmpFieldName, pThis->mTmpFieldValue);
			pThis->mTmpFieldValue = "";
			pThis->mTmpFieldName = "";
		}

		return 0;
	}
	int HttpParserImpl::onHeadersComplete(http_parser* parser) {
		HttpParserImpl* pThis = getThis(parser);
		pThis->mState = PARSE_HeaderComplete;
		INTERNAL_FFL_LOG_DEBUG("HttpParser: onHeadersComplete");
		return 0;
	}
	int HttpParserImpl::onBody(http_parser* parser, const char* at, size_t length) {
		//HttpParserImpl* pThis = getThis(parser);
		//FFL_LOG_DEBUG("HttpParser: onBody");
		return 0;
	}
	int HttpParserImpl::onMessageComplete(http_parser* parser) {
		HttpParserImpl* pThis = getThis(parser);
		pThis->mState = PARSE_COMPLETE;
		//FFL_LOG_DEBUG("HttpParser: onMessageComplete");		
		return 0;
	}



}

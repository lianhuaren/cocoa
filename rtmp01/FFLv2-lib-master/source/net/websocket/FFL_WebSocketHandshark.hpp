/*
*  This file is part of FFL project.
*
*  The MIT License (MIT)
*  Copyright (C) 2017-2018 zhufeifei All rights reserved.
*
*  FFL_WebSocketHandshark.hpp
*  Created by zhufeifei(34008081@qq.com) on 2019/02/25
*  https://github.com/zhenfei2016/FFLv2-lib.git
*
*  websocket握手
*/
#ifndef _FFL_WEBSOCKET_HANDSHARK_HPP_
#define _FFL_WEBSOCKET_HANDSHARK_HPP_

#include <net/http/FFL_HttpRequest.hpp>
#include <net/http/FFL_HttpResponse.hpp>
#include <net/http/FFL_HttpClient.hpp>

namespace FFL {	
	/////////////////////////////////////////////////////////////////////////
	
	class WSHandsharkRequest : public HttpRequest{
	public:
		WSHandsharkRequest(FFL::sp<HttpClient>& client);
		~WSHandsharkRequest();

		//
		//  获取设置版本号
		//		
		int32_t getSecVersion();
		//
		//  获取设置key
		//		
		bool getSecKey(String& key);

		//
		//  请求内容
		//  header :头内容		
		//  content:内容
		//
		virtual bool writeHeader();		
	protected:
		int32_t mVersion;
		String mKey;
	};

	class WSHandsharkResponse : public HttpResponse {
	public:
		//
		//  服务端应答创建函数
		//
		WSHandsharkResponse(FFL::sp<HttpClient>& client,String& requestKey);
		//
		//  接收端应答创建函数
		//
		WSHandsharkResponse(FFL::sp<HttpClient>& client);
		~WSHandsharkResponse();

		//
		//  获取服务端计算的key
		//		
		bool getSecAccess(String& key);
		//
		//  应答内容
		//  header :头内容		
		//  content:内容
		//  服务端应答调用此函数
		//
		virtual bool writeHeader();
	private:
		void calcSecAccess(const String& requestkey);
	private:
		String mSecAccess;
	public:		
		static bool isWebSocket(HttpResponse* response);
		static bool getSec_WebSocket_Server(HttpResponse* response, String& value);
		static bool getSec_WebSocket_Accept(HttpResponse* response,String& key);
		static bool isHandsharkOk(WSHandsharkRequest* request,HttpResponse* response);
	};

	//
	//  是否一个有效的握手请求
	//
	bool WebSocket_isHandsharkRequest(HttpRequest* request);
	bool WebSocket_getSecWebSocketkey(HttpRequest* request,String& key);
}

#endif

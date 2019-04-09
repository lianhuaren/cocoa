/*
*  This file is part of FFL project.
*
*  The MIT License (MIT)
*  Copyright (C) 2017-2018 zhufeifei All rights reserved.
*
*  FFL_WebSocketClient.cpp
*  Created by zhufeifei(34008081@qq.com) on 2019/02/24
*  https://github.com/zhenfei2016/FFLv2-lib.git
*
*  websocket客户端
*/

#include <net/websocket/FFL_WebSocketClient.hpp>
#include <net/http/FFL_HttpRequest.hpp>
#include <net/http/FFL_HttpResponse.hpp>
#include <net/FFL_NetEventLoop.hpp>
#include "FFL_WebSocketHandshark.hpp"
#include "FFL_WebSocketFrame.hpp"
#include "FFL_WebSocket.hpp"

namespace FFL {
	static void xorBytes(const uint8_t *in1, uint8_t *out, uint8_t maker[4], int size) {
		for (int i = 0; i < size; i++) {
			*out = in1[i] ^ maker[i % 4];
			out++;
		}
	}

	WebSocketClient::WebSocketClient() :
		mClient(NULL),
		mStream(NULL),
		mIsConnected(false),
		mIsHandShark(false),
		mWebSocket(NULL){
	}
	WebSocketClient::~WebSocketClient() {	
		close();		
	}
	//
	//  连接host
	//
	bool WebSocketClient::connect(const char* host) {	
		mHost = host;
		return connect("127.0.0.1",80);
	}
	bool WebSocketClient::connect(const char* ip, uint16_t port) {
		if (isConnected()) {
			return false;
		}

		TcpClient* client = new TcpClient();
		if (TcpClient::connect(ip,port, *client) != FFL_OK) {
			FFL_SafeFree(client);
			return false;
		}

		mClient = client;
		mIsConnected = true;
		if (mHost.isEmpty()) {
			mHost = String::format("%s:%d", ip, port);
		}
		return true;
	}
	bool WebSocketClient::isConnected() const {
		return mIsConnected;
	}
	NetFD WebSocketClient::getFd() {
		return mClient->getFd();
	}
	//
	//  关闭连接
	//
	void WebSocketClient::close() {	
		if (mIsConnected) {
			mClient->close();
			mIsConnected = false;
			FFL_SafeFree(mClient);
		}

		FFL_SafeFree(mWebSocket);
		mIsHandShark = false;
		FFL_SafeFree(mStream);
	}
	//
	//  发送握手请求，并等待握手成功应答
	//
	bool WebSocketClient::handshark(const char* path) {
		if (isHandshark()) {
			return false;
		}

		//
		// 发送握手请求
		//
		sp<HttpClient> httpClient = new HttpClient(mClient);
		FFL::HttpHeader header;		
		header.setKey("Host", mHost.string());				
		FFL::sp<WSHandsharkRequest> request = new FFL::WSHandsharkRequest(httpClient);
		request->setHeader(header);
		if (!request->send()) {
			return false;
		}

		//
		//  等待握手应答
		//		
		FFL::sp<HttpResponse> response=httpClient->readResponse();
		if (response.isEmpty()) {
			return false;
		}		
		if (!WSHandsharkResponse::isWebSocket(response.get())) {
			return false;
		}
		//
		//  验证key是否正确
		//
		if(!WSHandsharkResponse::isHandsharkOk(request.get(),response.get())){
			return false;
		}

		mStream = new NetStreamReader(mClient);

		uint8_t key[4] = {1,2,3,4};
		mWebSocket = new WebSocket(mClient, mStream, true, key);
		mIsHandShark = true;		
		return true;
	}
	bool WebSocketClient::isHandshark() const {
		return mIsHandShark;
	}	
	//
	//  接收一帧数据
	//  buffer: 缓冲区 ， 
	//  bufferSize：这个输入缓冲区的大小。 当成功接收数据的情况下返回接收的数据大小
	//
	bool WebSocketClient::recvFrame(uint8_t* buffer, uint32_t* bufferSize) {
		if (!isHandshark()) {
			return false;
		}

		return mWebSocket->recvFrame(buffer, bufferSize);
	}

	IOReader* WebSocketClient::createInputStream() {
		if (!isHandshark()) {
			return NULL;
		}
		return mWebSocket->createInputStream();
	}
	void WebSocketClient::destroyInputStream(IOReader* reader) {
		if (!isHandshark()) {
			return ;
		}
		mWebSocket->destroyInputStream(reader);
	}


	bool WebSocketClient::sendText(const char* text) {		
		if (!isHandshark()) {
			return false;
		}

		return mWebSocket->sendText(text);
	}
	bool WebSocketClient::sendBinary(const uint8_t* data, uint32_t len) {
		if (!isHandshark()) {
			return false;
		}

		return mWebSocket->sendBinary(data,len);
	}
	//
	//  读写二进制流
	//
	IOWriter* WebSocketClient::createOutputStream(uint32_t size) {		
		if (!isHandshark()) {
			return NULL;
		}

		return mWebSocket->createOutputStream(size);
	}
	void WebSocketClient::destroyOutputStream(IOWriter* writer) {
		if (!isHandshark()) {
			return ;
		}

		mWebSocket->destroyOutputStream(writer);
	}
	bool WebSocketClient::sendPing() {
		if (!isHandshark()) {
			return false;
		}

		return mWebSocket->sendPing();
	}
	bool WebSocketClient::sendPong() {
		if (!isHandshark()) {
			return false;
		}

		return mWebSocket->sendPong();
	}
	bool WebSocketClient::sendBye() {
		if (!isHandshark()) {
			return false;
		}

		return  mWebSocket->sendBye();
	}	
}

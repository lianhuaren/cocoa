/*
*  This file is part of FFL project.
*
*  The MIT License (MIT)
*  Copyright (C) 2017-2018 zhufeifei All rights reserved.
*
*  FFL_TcpClient.cpp
*  Created by zhufeifei(34008081@qq.com) on 2018/07/14
*  https://github.com/zhenfei2016/FFLv2-lib.git
*  网络客户端定义
*
*/

#include <net/FFL_TcpClient.hpp>

namespace FFL {
	
    //
    //  tcp相关的，连接到服务器上，如果设置过fd则返回false
    //
    status_t TcpClient::connect(const char* ip, uint16_t port,TcpClient& client){        
        return  client.mSocket.connect(ip,port)?FFL_OK : FFL_FAILED;
    }   

    TcpClient::TcpClient(NetFD fd):mPriv(NULL){
		mSocket.setFd(fd, CSocket::PROTOCOL_TCP);
    }
    TcpClient::~TcpClient(){		
    }
	//
	// 保存一些用户数据
	//
	void TcpClient::setUserdata(void* priv) {
		mPriv = priv;
	}
	void* TcpClient::getUserdata() {
		return mPriv;
	}

	NetFD TcpClient::getFd() {
		return mSocket.getFd();
	}
    //
    //  读数据到缓冲区
    //  buf:缓冲区地址
    //  count:需要读的大小
    //  pReaded:实质上读了多少数据
    //  返回错误码  ： FFL_OK表示成功
    //
    status_t TcpClient::read(uint8_t* buf, size_t count, size_t* pReaded){
        return  mSocket.read(buf,count,pReaded);
    }
    //
    //  写数据到文件中
    //  buf:缓冲区地址
    //  count:缓冲区大小
    //  pWrite:实质上写了多少数据
    //  返回错误码  ： FFL_OK表示成功
    //
    status_t TcpClient::write(const void* buf, size_t count, size_t* pWrite){
		return  mSocket.write(buf, count, pWrite);
    }
    //
    //  写数据到文件中
    //  bufVec:缓冲区地址,数组
    //  count:数组大小
    //  pWrite:实质上写了多少数据
    //  返回错误码  ： FFL_OK表示成功
    //
    status_t TcpClient::writeVec(const BufferVec* bufVec, int count, size_t* pWrite){
		return  mSocket.writeVec(bufVec, count, pWrite);
    }
	void TcpClient::close() {
		mSocket.close();
	}

	
	
}

/*
*  This file is part of FFL project.
*
*  The MIT License (MIT)
*  Copyright (C) 2017-2018 zhufeifei All rights reserved.
*
*  FFL_UdpServer.hpp
*  Created by zhufeifei(34008081@qq.com) on 2018/07/14
*  https://github.com/zhenfei2016/FFLv2-lib.git
*  udp服务器
*
*/
#ifndef _FFL_UDP_SERVER_HPP_
#define _FFL_UDP_SERVER_HPP_

#include <net/FFL_NetEventLoop.hpp>
#include <net/FFL_UdpClient.hpp>

namespace FFL {    
	class UdpServerImpl;
	class FFLIB_API_IMPORT_EXPORT UdpServer : public Module{
	public:
		class Callback {
		public:
			virtual bool onClientReceived(UdpClient* client,const char* packet,size_t size) = 0;
		};
	public:
        //
        //   ip:服务地址
        //   port: 服务端口
        //   handler ：处理句柄
        //   name:  服务名称
        //
		UdpServer(const char* ip,uint16_t port,
			UdpServer::Callback* handler,
			      const char* name=NULL);
		virtual ~UdpServer();
		//
		//   阻塞的线程中执行的eventloop,返回是否继续进行eventLoop
		//   waitTime:输出参数，下一次执行eventLoop等待的时长
		//   true  : 继续进行下一次的eventLoop
		//   false : 不需要继续执行eventloop
		//
		virtual bool eventLoop(int32_t* waitTime);
	protected:
		//
		//  调用。start，stop会触发onStart,onStop的执行
		//  onStart :表示准备开始了 ,可以做一些初始化工作
		//  onStop :表示准备停止了 ,可以做停止前的准备，想置一下信号让eventloop别卡住啊 
		//  在这些函数中，不要再调用自己的函数，例如：start,stop, isStarted等
		//
		virtual bool onStart();
		virtual void onStop();	
	protected:
		UdpServerImpl* mImpl;
	};
}
#endif

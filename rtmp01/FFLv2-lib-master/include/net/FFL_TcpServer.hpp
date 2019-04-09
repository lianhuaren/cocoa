/*
*  This file is part of FFL project.
*
*  The MIT License (MIT)
*  Copyright (C) 2017-2018 zhufeifei All rights reserved.
*
*  FFL_TcpServer.cpp
*  Created by zhufeifei(34008081@qq.com) on 2018/07/14
*  https://github.com/zhenfei2016/FFLv2-lib.git
*  tcp服务器
*
*/

#ifndef _FFL_SIMPLE_TCPSERVER_HPP_
#define _FFL_SIMPLE_TCPSERVER_HPP_

#include <net/FFL_NetEventLoop.hpp>
#include <net/FFL_TcpListener.hpp>
#include <net/FFL_TcpClient.hpp>

namespace FFL {    
	class TcpServerImpl;
	class FFLIB_API_IMPORT_EXPORT TcpServer : public Module{
	public:
		class Callback {
		public:
			enum FD_OPTMODE {
				//
				// 继续数据的读写
				FD_CONTINUE = 1,
				//
				// 结束关闭fd
				FD_DESTROY = 2,
				//
				//  fd事件管理中移除出去，自己进行TcpClient的读写
				FD_REMOVE = 4,
			};

			//
			//  aliveTimeUs:保活时长，如果超过这么长时间还没有数据则干掉这个client
			//              <0 一直存活， 
			//
			virtual bool onClientCreate(TcpClient* fd,int64_t* aliveTimeUs) = 0;
			virtual void onClientDestroy(TcpClient* fd, FD_OPTMODE mod) = 0;
			//
			// 收到网络数据后，返回对这个client的处理方式   
			//			
			virtual FD_OPTMODE onClientReceived(TcpClient* fd) = 0;

		};
	public:
        //
        //   ip:服务地址
        //   port: 服务端口
        //   handler ：处理句柄
        //   name:  服务名称
        //
		TcpServer(const char* ip,uint16_t port,
			      TcpServer::Callback* handler,
			      const char* name=NULL);
		virtual ~TcpServer();

		
		//
		//   阻塞的线程中执行的eventloop,返回是否继续进行eventLoop
		//   waitTime:输出参数，下一次执行eventLoop等待的时长
		//   true  : 继续进行下一次的eventLoop
		//   false : 不需要继续执行eventloop
		//
		virtual bool eventLoop(int32_t* waitTime) ;
	protected:
		virtual bool onStart();
		virtual void onStop();
	private:
		TcpServerImpl* mImpl;
	};
}
#endif

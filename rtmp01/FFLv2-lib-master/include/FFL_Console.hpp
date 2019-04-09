/*
*  This file is part of FFL project.
*
*  The MIT License (MIT)
*  Copyright (C) 2017-2018 zhufeifei All rights reserved.
*
*  FFL_Console.hpp
*  Created by zhufeifei(34008081@qq.com) on 2019/03/11
*  https://github.com/zhenfei2016/FFL-v2.git
*  命令行处理的帮助类
*  
*/
#ifndef _FFL_CONSOLE_HPP_
#define _FFL_CONSOLE_HPP_

#include <FFL_Console.h>
#include <FFL_Module.hpp>

namespace FFL {
	class FFLIB_API_IMPORT_EXPORT ConsoleLoop : public Module {
	public:
		ConsoleLoop();
		virtual ~ConsoleLoop();
		//
		//  注册处理命令，必须在start前进行注册的
		//  cmdTable:命令数组
		//  size :命令个数
		//  最多支持100个命令
		//
		bool registeCommand(CmdOption* cmdTable,uint32_t size);		
		//
		//  打印用法
		//
		void dumpUsage();
		//
		// 用于透传的用户数据
		//
		void setUserdata(void* userdata);
		void* getUserdata() const;
	public:
		//
		//   阻塞的线程中执行的eventloop,返回是否继续进行eventLoop
		//   waitTime:输出参数，下一次执行eventLoop等待的时长
		//   true  : 继续进行下一次的eventLoop
		//   false : 不需要继续执行eventloop
		//
		virtual bool eventLoop(int32_t* waitTime);	
	private:		
		CmdOption* mRegistedCommand;
		void* mUserData;
	};
}

#endif
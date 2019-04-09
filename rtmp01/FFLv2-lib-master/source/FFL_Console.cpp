/*
*  This file is part of FFL project.
*
*  The MIT License (MIT)
*  Copyright (C) 2017-2018 zhufeifei All rights reserved.
*
*  FFL_Console.cpp
*  Created by zhufeifei(34008081@qq.com) on 2019/03/11
*  https://github.com/zhenfei2016/FFL-v2.git
*  命令行处理的帮助类
*/
#include <FFL_Console.hpp>

namespace FFL {
	static int QuitFlag(void* userdata) {
		ConsoleLoop* loop = static_cast<ConsoleLoop*>(userdata);
		return loop->isStarted()?0:1;
	}

	const static int KMaxCmdArraySize=101;
	ConsoleLoop::ConsoleLoop():mUserData(NULL){
		mRegistedCommand = new CmdOption[KMaxCmdArraySize];
		memset(mRegistedCommand, 0, sizeof(CmdOption)*KMaxCmdArraySize);
	}
	ConsoleLoop::~ConsoleLoop() {
		for (uint32_t i = 0; i < KMaxCmdArraySize; i++) {
			CmdOption* cmd = &mRegistedCommand[i];
			FFL_free((void*)cmd->mName);
			FFL_free((void*)cmd->nHelp);
		}
		FFL_SafeFreeA(mRegistedCommand);
	}
	//
	//  注册处理命令，必须在start前进行注册的
	//  cmdTable:命令数组
	//  size :命令个数
	//
	bool ConsoleLoop::registeCommand(CmdOption* cmdTable, uint32_t size) {
		CmdOption* pCmd = NULL;		 
		for (uint32_t i = 0; i < KMaxCmdArraySize; i++) {
			CmdOption* cmd = &mRegistedCommand[i];
			if (cmd->mName != NULL) {
				continue;
			}

			if (KMaxCmdArraySize - i < size) {
				return false;
			}

			pCmd = cmd;
			break;
		}

		for (uint32_t i=0; i < size; i++) {
			CmdOption* cmd = &pCmd[i];
			cmd->mName = FFL_strdup(cmdTable[i].mName);
			cmd->nHelp = FFL_strdup(cmdTable[i].nHelp);
			cmd->mHaveAargument = cmdTable[i].mHaveAargument;
			cmd->fun = cmdTable[i].fun;
		}
		return true;
	}
	//
	//  打印用法
	//
	void ConsoleLoop::dumpUsage(){
		FFL_printUsage(mRegistedCommand);
	}
	//
	// 用于透传的用户数据
	//
	void ConsoleLoop::setUserdata(void* userdata) {
		mUserData = userdata;
	}
	void* ConsoleLoop::getUserdata() const {
		return mUserData;
	}
	//
	//   阻塞的线程中执行的eventloop,返回是否继续进行eventLoop
	//   waitTime:输出参数，下一次执行eventLoop等待的时长
	//   true  : 继续进行下一次的eventLoop
	//   false : 不需要继续执行eventloop
	//
	bool ConsoleLoop::eventLoop(int32_t* waitTime) {
		FFL_consoleEvenLoop(mRegistedCommand, this, QuitFlag);
		return false;
	}
}

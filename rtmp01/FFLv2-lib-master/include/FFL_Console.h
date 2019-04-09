/*
*  This file is part of FFL project.
*
*  The MIT License (MIT)
*  Copyright (C) 2017-2018 zhufeifei All rights reserved.
*
*  FFL_Console.h   
*  Created by zhufeifei(34008081@qq.com) on 2018/07/21 
*  https://github.com/zhenfei2016/FFL-v2.git
*  命令行处理的帮助类
*
*   static void play(const char* args, void* userdata) {
*	}
*
*
*	static void stop(const char* args, void* userdata) {
*	}
*   static CmdOption  myCmdOption[] = {
*	   { "play",0,play,"play video " },
*	   { "stop",0,stop,"stop video " },
*	   { 0,0,0,0 }
*	};
*
*	int QuitFlag(void* userdata){
*	   return 0;
*	}
*
*	main(){
*	   FFL_inputLooper(myCmdOption,0,QuitFlag);
*	}
*
*
*/

#ifndef _FFL_COMMANDHELPER_H_
#define _FFL_COMMANDHELPER_H_

#include <FFL_Core.h>
#ifdef  __cplusplus
extern "C" {
#endif
	/*
	* 命令行参数选项
	*/
	typedef struct CmdOption {
		/*
		*  命令名称
		*/
		const char* mName;
		/*
		*  是否存在参数 0 , 1
		*/
		int32_t  mHaveAargument;
		/*
		*  这个命令需要执行的函数
		*/
		void(*fun)(const char* value, void* userdata);
		/*
		*  帮助信息
		*/
		const char* nHelp;
	}CmdOption;

	/*
	*   分析命令行
	*   argc：参数个数
	*   argv：参数数组
	*   opts：支持的命令数组，以null结尾
	*   size：opts数组的大少
	*   userdata: 透传到CmdOption中的函数回调中
	*   命令行格式  --cmd=12344
	*   返回命令在opts中的位置，没找到返回-1
	*/
	FFLIB_API_IMPORT_EXPORT int FFL_parseCommnadline(int argc, const char** argv, CmdOption* opts, int size, void* userdata);
	/*
	*  fnQuitFlag 返回非0则退出循环
	*/
	FFLIB_API_IMPORT_EXPORT void FFL_consoleEvenLoop(CmdOption* opts, void* userdata,
		    int(*fnQuitFlag)(void* userdata) );

	/*
	*  打印帮助
	*/
	FFLIB_API_IMPORT_EXPORT void FFL_printUsage(CmdOption* opts );

#ifdef  __cplusplus
}
#endif


#endif
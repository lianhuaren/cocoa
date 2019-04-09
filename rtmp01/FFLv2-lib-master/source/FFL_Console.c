/*
*  This file is part of FFL project.
*
*  The MIT License (MIT)
*  Copyright (C) 2017-2018 zhufeifei All rights reserved.
*
*  FFL_Console.c
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
*	   FFL_consoleEvenLoop(myCmdOption,0,QuitFlag);
*	}
*
*/
#include <FFL_Console.h>

#include "getopt/getopt.h"

static struct option* alloc_getopt_Option(CmdOption* opts, int size) {
	struct option* optionList = FFL_mallocz((size + 1) * sizeof(struct option));
	int i = 0;
	for (i = 0; i < size; i++) {
		if (!opts[i].mName) {
			optionList[i].name = FFL_strdup("__dummy__");
			continue;
		};

		optionList[i].name = FFL_strdup(opts[i].mName);
		optionList[i].has_arg = optional_argument;//(opts[i].mHaveAargument !=0)? optional_argument : no_argument;
		optionList[i].flag = NULL;
		optionList[i].val = 0;
	}
	return optionList;
}

static void free_getopt_Option(struct option* opts) {
	struct option* opt = opts;
	while (opt->name)
	{
		FFL_free((void *)opt->name);
		opt++;
	}

	FFL_free(opts);
}

/*
*   分析命令行
*   argc：参数个数
*   argv：参数数组
*   opts：支持的命令数组，以null结尾
*   size：opts数组的大少
*   命令行格式  --cmd=12344
*   返回命令在opts中的位置，没找到返回-1
*/
int FFL_parseCommnadline(int argc, const char** argv, CmdOption* opts, int size, void* userdata) {
	struct option* longOpt = alloc_getopt_Option(opts, size);

	char* params = 0;
	int   paramsLen = 0;

	int cmdIndex = -1;
	int optionIndex = -1;
	int opt = -1;
	optind = 1;
	while ((opt = getopt_long_only(argc,
		(char *const *)argv,
		"",
		longOpt,
		&optionIndex)) != -1)
	{
		if (optionIndex != -1) {
			cmdIndex = optionIndex;
			if (opts[cmdIndex].fun) {
				/*
				*  参数拷贝一次，去掉最后边的回车换行
				*/
				paramsLen = 0;
				params = 0;
				if (optarg != NULL) {
					paramsLen = strlen(optarg);
				}
				if (paramsLen > 1) {
					params = FFL_mallocz(paramsLen + 1);
					memcpy(params, optarg, paramsLen);
					if (params[paramsLen - 1] == '\n' || params[paramsLen - 1] == '\r') {
						params[paramsLen - 1] = 0;
					}
				}

				opts[cmdIndex].fun(params, userdata);

				FFL_free(params);
				params = 0;
				paramsLen = 0;
			}
		}
		optionIndex = -1;
	}
	free_getopt_Option(longOpt);

	return cmdIndex;
}


static void help(const char* args, void* userdata);
static CmdOption  gCmdOption[] = {
	{ "help",0,help,"printf help" },
	{ 0,0,0,0 }
};

static void help(const char* args, void* userdata) {
	int32_t i = 0;
	CmdOption* opts =(CmdOption*) userdata;	
    while(opts->mName){
		printf("(user func) %s  :%s \n", opts->mName, opts->nHelp? opts->nHelp:"");
		opts++;
	}	
	i = 0;
	opts = gCmdOption;
	while (opts->mName) {
		printf("(sys  func) %s  :%s \n", opts->mName, opts->nHelp ? opts->nHelp : "");
		opts++;
	}
	printf("----------------------------------------------------------------------\n");
}

/*
*  fnQuitFlag 返回非0则退出循环
*/
void FFL_consoleEvenLoop(CmdOption* opts,void* userdata,
	int(*fnQuitFlag)(void* userdata)){
	char cmd[256] = {0};
	const char *argv[] = {
		"",
		cmd,
	};
	int argc = 2;
	int optCount = 0; 
    
	CmdOption* opt;
	char* pCmdLine =0;

	cmd[0] = '-';
	cmd[1] = '-';

	{
		opt = opts;		
		while (opt->mName) {
			optCount++;
			opt++;
		}
	}

	while (fgets(cmd + 2, 256 - 3, stdin)) {
		//
		//  把输入命令格式化为  cmd=xxx  ,就是把命令转化成第一个参数
		//		
		pCmdLine = cmd + 2;
		while (*pCmdLine++) {
			if (*pCmdLine == ' ' || *pCmdLine == '\n' || *pCmdLine == '\r') {
				*pCmdLine = '=';
				break;
			}
		}

		if (FFL_parseCommnadline(argc, argv,
			opts,
			optCount,
			userdata) < 0) {
			/*
			*  看一下本系统是否支持这个命令
			*/
			if(FFL_parseCommnadline(argc, argv,
				gCmdOption,
				FFL_ARRAY_ELEMS(gCmdOption),
				opts) < 0){

				*pCmdLine = 0;
				printf("\"%s\"  Not an command. \n", cmd + 2);
			}		
		}

		if (fnQuitFlag && fnQuitFlag(userdata)!=0) {
			break;
		}
	}
}
/*
*  打印帮助
*/
void FFL_printUsage(CmdOption* opts) {
	help(0,opts);
}

/*
*  This file is part of FFL project.
*
*  The MIT License (MIT)
*  Copyright (C) 2017-2018 zhufeifei All rights reserved.
*
*  FFL_Log
*  Created by zhufeifei(34008081@qq.com) on 2017/7/12
*  https://github.com/zhenfei2016/FFLv2-lib.git
*  打印日志帮助类
*
*/
#include <stdio.h>
#include <stdarg.h>

#include <FFL_Log.h>
#include <FFL_Time.h>
#include <FFL_Thread.h>

/*
 *  默认情况下打印的最大日志等级
 *  debug模式下打印所有日志
 */
#ifdef _DEBUG
static FFL_LogLevel gLogMaxLevel = FFL_LOG_LEVEL_ALL;
#else
static FFL_LogLevel gLogMaxLevel = FFL_LOG_LEVEL_ERROR;
#endif 

/*
 *  默认情况下打印日志函数
 */
static int defaultPrintLog(int level,const char* tag, const char *format, va_list vl);
static FFL_LogHookFun gLogHookFun = NULL;
static void* gLogHookFunUserdata = 0;
/*
 * 默认日志输出到这个fd上
*/
static FILE *gDefaultLogFd=0;
/*
 *  所有的打印函数入口
 */
static void internalPrintLog(int level,const char* tag,int enableHook, const char *format,va_list args){
	if (level <= gLogMaxLevel) {
		if (enableHook && gLogHookFun && (gLogHookFun(level, tag, format, args, gLogHookFunUserdata) > 0)) {
			return;
		}

		/*
		 * 默认日志处理函数
		 */
		defaultPrintLog(level, tag, format, args);
	}
}

static const char *s_levels_string[] = {
  "CRI",
  "ERR",
  "WAR",
  "INF",
  "DBG"
};

void FFL_LogSetLevel(FFL_LogLevel level)
{
	gLogMaxLevel = level;
}
FFL_LogLevel FFL_LogGetLevel()
{
	return gLogMaxLevel;
}
const char* FFL_LogGetLevelString(int level) {
	if (level < 0 || level >= FFL_LOG_LEVEL_ALL) {
		return "NUL";
	}

	return s_levels_string[level];
}

void FFL_LogHook(FFL_LogHookFun callback, void* userdata){
	gLogHookFun = callback;
	gLogHookFunUserdata = userdata;
}

/*
*  每条的输出日志日志最长
*  调用这个函数的时候需要保证level值是一个有效值
*/
#define MAX_LOG_LENGTH	4096
static int defaultPrintLog(int level,const char* tag, const char *format, va_list vl){
	char str[MAX_LOG_LENGTH] = {0};
	char timeFormat[64] = { 0 };	
	
	vsnprintf(str, MAX_LOG_LENGTH - 1, format, vl);
	if ( !gDefaultLogFd) 
		gDefaultLogFd = stdout;
	
	{
		FFL_getNowString(timeFormat,63);
#ifdef WIN32		
		fprintf(gDefaultLogFd, "%s:%s tid=%d %s \n", s_levels_string[level], timeFormat, (int)FFL_CurrentThreadID(),  str);	
#else
        fprintf(gDefaultLogFd, "%s:%s tid=%d %s \n", s_levels_string[level], timeFormat, (int)FFL_CurrentThreadID(),  str);
#endif
#ifdef _DEBUG
		fflush(gDefaultLogFd);
#endif
	}
	return 1;
}
/*
* 打印日志
*/
FFLIB_API_IMPORT_EXPORT int FFL_LogPrint(int level, const char *format, ...) {
	if (level >= 0 && level < gLogMaxLevel) {
		va_list args;
		va_start(args, format);
		internalPrintLog(level, 0,1, format, args);
		va_end(args);
		return 0;
	}
	return -1;
}
FFLIB_API_IMPORT_EXPORT int FFL_LogPrintTag(int level, const char *tag, const char *format, ...) {
	if (level >= 0 && level < gLogMaxLevel) {
		va_list args;
		va_start(args, format);
		internalPrintLog(level, tag, 1, format, args);
		va_end(args);
		return 0;
	}
	return -1;
}
FFLIB_API_IMPORT_EXPORT int FFL_LogPrintV(int level, const char *format, va_list args) {
	if (level >= 0 && level < gLogMaxLevel) {		
		internalPrintLog(level, 0, 1, format, args);
		return 0;
	}
	return -1;
}
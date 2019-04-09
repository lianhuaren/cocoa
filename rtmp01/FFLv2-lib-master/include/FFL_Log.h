/*
*  This file is part of FFL project.
*
*  The MIT License (MIT)
*  Copyright (C) 2017-2018 zhufeifei All rights reserved.
*
*  FFL_Log   
*  Created by zhufeifei(34008081@qq.com) on 2017/7/12
*  https://github.com/zhenfei2016/FFLv2-lib.git
*
*  打印日志帮助类，
*  增加支持的编译日志等级  2017/12/22
*  FFLIB_COMPILE_LOG_LEVEL = 3  对应FFL_LogLevel的值 
*
*/
#ifndef __FFL_LOG_H__
#define __FFL_LOG_H__

#include <stdio.h>
#include <stdarg.h>
#include <FFL_Config.h>

#ifdef  __cplusplus
extern "C" {
#endif	

/************************************************************************
*	支持的日志等级
***********************************************************************/
typedef enum
{
	FFL_LOG_LEVEL_CRIT = 0,
	FFL_LOG_LEVEL_ERROR,
	FFL_LOG_LEVEL_WARNING,
	FFL_LOG_LEVEL_INFO,
	FFL_LOG_LEVEL_DEBUG,
	FFL_LOG_LEVEL_ALL
} FFL_LogLevel;
/************************************************************************
* 设置日志等级
***********************************************************************/
FFLIB_API_IMPORT_EXPORT void FFL_LogSetLevel(FFL_LogLevel level);
FFLIB_API_IMPORT_EXPORT FFL_LogLevel FFL_LogGetLevel();
FFLIB_API_IMPORT_EXPORT const char* FFL_LogGetLevelString(int level);

/************************************************************************
*  设置日志输出到哪里去，外部接管，还是指定文件中
***********************************************************************/
/*
 *  接管日志输出函数
 *  根据返回值表示是否需要继续默认的日志输出
 *  1 : 已经把日志处理了，不需要默认日志系统了
 *  0 : 用默认日志处理函数    
 *  userdata :透传数据
 */
typedef int (*FFL_LogHookFun)(int level,const char* tag,const char *format, va_list args,void* userdata);
FFLIB_API_IMPORT_EXPORT void FFL_LogHook(FFL_LogHookFun callback, void* userdata);
/************************************************************************
  通用输出，特定等级输出，特定等级并且带tag的输出
  返回0表示成功

  默认的情况下一条日志的最大长度为4k
************************************************************************/
FFLIB_API_IMPORT_EXPORT int FFL_LogPrint(int level, const char *format, ...);
FFLIB_API_IMPORT_EXPORT int FFL_LogPrintTag(int level, const char *tag, const char *format, ...);
FFLIB_API_IMPORT_EXPORT int FFL_LogPrintV(int level, const char *format, va_list args);

#ifdef  __cplusplus
}
#endif
#endif

/*
*  This file is part of FFL project.
*
*  The MIT License (MIT)
*  Copyright (C) 2017-2018 zhufeifei All rights reserved.
*
*  FFL_LogConfig.h   
*  Created by zhufeifei(34008081@qq.com) on 2019/1/4
*  https://github.com/zhenfei2016/FFLv2-lib.git
*
*  cmake生成的日志宏
*
*/
#ifndef __FFL_LOG_CONFIG_H__
#define __FFL_LOG_CONFIG_H__

#define FFL_LOG_CRIT(format,...) FFL_LogPrint(FFL_LOG_LEVEL_CRIT,format,##__VA_ARGS__)
#define FFL_LOG_CRIT_TAG(tag,format,...) FFL_LogPrintTag(FFL_LOG_LEVEL_CRIT,tag,format,##__VA_ARGS__)

#define FFL_LOG_ERROR(format,...) FFL_LogPrint(FFL_LOG_LEVEL_ERROR,format,##__VA_ARGS__)
#define FFL_LOG_ERROR_TAG(tag,format,...) FFL_LogPrintTag(FFL_LOG_LEVEL_ERROR,tag,format,##__VA_ARGS__)

#define FFL_LOG_WARNING(format,...) FFL_LogPrint(FFL_LOG_LEVEL_WARNING,format,##__VA_ARGS__)
#define FFL_LOG_WARNING_TAG(tag,format,...) FFL_LogPrintTag(FFL_LOG_LEVEL_WARNING,tag,format,##__VA_ARGS__)

#define FFL_LOG_INFO(format,...) FFL_LogPrint(FFL_LOG_LEVEL_INFO,format,##__VA_ARGS__) 
#define FFL_LOG_INFO_TAG(tag,format,...) FFL_LogPrintTag(FFL_LOG_LEVEL_INFO,tag,format,##__VA_ARGS__)

#define FFL_LOG_DEBUG(format,...) 
#define FFL_LOG_DEBUG_TAG(tag,format,...) 

#endif

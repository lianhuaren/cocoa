/*
*  This file is part of FFL project.
*
*  The MIT License (MIT)
*  Copyright (C) 2017-2018 zhufeifei All rights reserved.
*
*  FFL_Path.h   
*  Created by zhufeifei(34008081@qq.com) on 2018/07/25 
*  https://github.com/zhenfei2016/FFL-v2.git
*
*  路径相关的处理类
*/
#ifndef __FFL_PATH_H__
#define __FFL_PATH_H__

#include <FFL_Core.h>
#ifdef  __cplusplus
extern "C" {
#endif
	/*
	 * 获取当前进程的路径，名称
	 * processdir : 返回进程路径，必需不是空的
	 * len ：buf的大小
	 * processname ： 如果非空，则返回进程名称
	 */
	FFLIB_API_IMPORT_EXPORT status_t FFL_getCurrentProcessPath(char* processdir, size_t len, char* processname);
    
    /*
     *  获取工作目录
     */
	FFLIB_API_IMPORT_EXPORT status_t FFL_getWorkPath(char* workPath, size_t len);
#ifdef  __cplusplus
}
#endif

#endif

/*
*  This file is part of FFL project.
*
*  The MIT License (MIT)
*  Copyright (C) 2017-2018 zhufeifei All rights reserved.
*
*  FFL_Path.c   
*  Created by zhufeifei(34008081@qq.com) on 2018/07/25 
*  https://github.com/zhenfei2016/FFL-v2.git
*
*  路径处理
*/
#include <FFL_Path.h>

#include <stdio.h>
#include <stdarg.h>
#include <limits.h>
#include <string.h>
#if WIN32 
#include <windows.h>
#elif defined(MACOSX)
#include <mach-o/dyld.h>
#else
#include <unistd.h>
#endif
/*
* 获取当前进程的路径，名称
*/
status_t FFL_getCurrentProcessPath(char* processdir, size_t len, char* processname) {
    char* pathEnd=NULL;
    uint32_t bufSize=(uint32_t)len;
    char separator='/';
#if WIN32
	GetModuleFileNameA(NULL, processdir, len);
    separator='\\';
#elif defined(MACOSX)
    _NSGetExecutablePath(processdir,&bufSize);
#else
	if (readlink("/proc/self/exe", processdir, len) <= 0) {
		return FFL_FAILED;
	}
#endif
    pathEnd = strrchr(processdir, separator);
    if(pathEnd) {
        ++pathEnd;
        if (processname) {
            strncpy(processname, pathEnd, len - 1);
		}
        *pathEnd = '\0';
    }
	return FFL_OK;
}

/*
 *  获取工作目录
 */
status_t FFL_getWorkPath(char* workPath, size_t len) {
    int32_t size=0;
    char separator='/';
#if WIN32
    GetCurrentDirectory(len, workPath);
    separator='\\';
#elif IOS
    return  FFL_FAILED;
#else
    getcwd(workPath,len);
    separator='/';
#endif
    size=strlen(workPath);
    if(size>0 && workPath[size-1]!=separator){
        workPath[size]=separator;
        workPath[size+1]=0;
    }
    
    return FFL_OK;
    
}


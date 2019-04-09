/*
*  This file is part of FFL project.
*
*  The MIT License (MIT)
*  Copyright (C) 2017-2018 zhufeifei All rights reserved.
*
*  sysFile.h
*  Created by zhufeifei(34008081@qq.com) on 2018/07/26
*  https://github.com/zhenfei2016/FFL-v2.git
*  文件操作，包装系统函数
*
*/
#ifndef _INETRNAL_FILE_H_
#define _INETRNAL_FILE_H_
#include <FFL.h>
#ifdef _WIN32
#include <windows.h>
#include <Shlwapi.h>
#pragma comment(lib,"shlwapi.lib")
#else
#include <unistd.h>
#include <sys/uio.h>
#endif
#include <fcntl.h>


typedef enum OpenFileMode {
	MODE_OPEN,
	MODE_APPEND,
	MODE_ALWAYS_CREATE,
}OpenFileMode;


typedef struct FileHandle {
#ifdef _WIN32
	HANDLE fd;
#else
	int fd;
#endif
}FileHandle;


#ifdef  __cplusplus
extern "C" {
#endif


	FileHandle* createFile(const char* path, OpenFileMode mode);
	void closeFile(FileHandle* fd);

	int writeFile(FileHandle* fd, void* data, int32_t size);
	int readFile(FileHandle* fd, uint8_t* buf, int32_t size);
#ifdef  __cplusplus
}
#endif


#endif
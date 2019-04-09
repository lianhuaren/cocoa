/*
*  This file is part of FFL project.
*
*  The MIT License (MIT)
*  Copyright (C) 2017-2018 zhufeifei All rights reserved.
*
*  sysFile.c   
*  Created by zhufeifei(34008081@qq.com) on 2018/07/26 
*  https://github.com/zhenfei2016/FFL-v2.git
*  文件操作，包装系统函数
*
*/
#include "sysFile.h"

#ifdef WIN32
FileHandle* createFile(const char* path, OpenFileMode mode) {
	DWORD openMode = OPEN_EXISTING;
	HANDLE fd=NULL;
	if (mode == MODE_OPEN) {
		openMode = OPEN_EXISTING;
	}
	else if (mode == MODE_ALWAYS_CREATE) {
		openMode = CREATE_ALWAYS;
	}
	else {

	}

	fd = CreateFileA(path,
		GENERIC_READ | GENERIC_WRITE,
		FILE_SHARE_WRITE | FILE_SHARE_READ,
		NULL,
		openMode,
		0, NULL);

	if (fd != INVALID_HANDLE_VALUE) {
		FileHandle* handle = FFL_mallocz(sizeof(FileHandle));
		handle->fd = fd;

		if (MODE_APPEND) {

		}
		return handle;
	}
	return NULL;
}
void closeFile(FileHandle* fd) {
	if (fd) {
		CloseHandle(fd->fd);
		FFL_free(fd);
	}
}

int writeFile(FileHandle* fd, void* data, int32_t size) {
	if (fd) {
		DWORD dwWrited = 0;
		if (WriteFile(fd->fd, data, size, &dwWrited, NULL)) {
			return (int)dwWrited;
		}
	}
	return 0;
}
int readFile(FileHandle* fd, uint8_t* buf, int32_t size) {
	if (fd) {
		DWORD dwReaded = 0;
		if (ReadFile(fd->fd, buf, size, &dwReaded, NULL)) {
			return (int)dwReaded;
		}
	}
	return 0;
}
#else
//
//定义flags:只写，文件不存在那么就创建，文件长度戳为0
//
#define FILE_FLAGS O_WRONLY | O_CREAT | O_TRUNC
//
//追加方式  O_APPEND
//

//
//创建文件的权限，用户读、写、执行、组读、执行、其他用户读、执行
//
#define FILE_MODE S_IRWXU | S_IXGRP | S_IROTH | S_IXOTH

FileHandle* createFile(const char* path, OpenFileMode mode) {
	int fd = -1;
	if (mode == MODE_OPEN) {
		fd = open(path, O_RDWR | O_CREAT);
	}
	else if (mode == MODE_APPEND) {
		fd = open(path, O_RDWR | O_APPEND);
	}
	else {
		fd = open(path, O_RDWR | O_CREAT | O_TRUNC, 777);
	}
	if (fd < 0) {
		return NULL;
	}
	FileHandle* handle = FFL_mallocz(sizeof(FileHandle));
	handle->fd = fd;
	return handle;
}
void closeFile(FileHandle* fd) {
	if (fd) {
		close(fd->fd);
		FFL_free(fd);
	}
}

int writeFile(FileHandle* fd, void* data, int32_t size) {
	if (fd && fd->fd >= 0) {
		int ret = write(fd->fd, data, size);
		if (ret < 0) {}

		return ret;
	}
	return 0;
}
int readFile(FileHandle* fd, uint8_t* buf, int32_t size) {
	if (fd  && fd->fd >= 0) {
		int ret = read(fd->fd, buf, size);
		if (ret < 0) {}
		return ret;
	}
	return 0;
}
#endif

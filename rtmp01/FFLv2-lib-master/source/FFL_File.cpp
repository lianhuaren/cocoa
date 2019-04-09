/*
*  This file is part of FFL project.
*
*  The MIT License (MIT)
*  Copyright (C) 2017-2018 zhufeifei All rights reserved.
*
*  FFL_File.cpp
*  Created by zhufeifei(34008081@qq.com) on 2018/06/20
*  https://github.com/zhenfei2016/FFLv2-lib.git
*  文件操作类
*
*/

#include <FFL_File.hpp>
#ifdef _WIN32
#include <windows.h>
#include <Shlwapi.h>
#pragma comment(lib,"shlwapi.lib")
#else
#include <unistd.h>
#include <sys/uio.h>
#endif
#include <fcntl.h>

enum OpenFileMode{
	MODE_OPEN,
	MODE_APPEND,
	MODE_ALWAYS_CREATE,
};


struct FileHandle{
#ifdef _WIN32
	HANDLE fd;
#else
	int fd;
#endif
};

#ifdef WIN32
FileHandle* createFile(const char* path, OpenFileMode mode) {
	DWORD openMode = OPEN_EXISTING;
	if (mode == MODE_OPEN) {
		openMode = OPEN_EXISTING;
	}else if (mode == MODE_ALWAYS_CREATE) {
		openMode = CREATE_ALWAYS;
	}else {

	}

	HANDLE h=::CreateFileA(path, 
		GENERIC_READ| GENERIC_WRITE, 
		FILE_SHARE_WRITE |FILE_SHARE_READ,
		NULL,
		openMode,
		NULL,NULL);

	if (h != INVALID_HANDLE_VALUE) {
		FileHandle* handle = new FileHandle();
		handle->fd = h;

		if (MODE_APPEND) {

		}
		return handle;
	}
	return NULL;
}
void closeFile(FileHandle* fd) {
	if (fd) {
		::CloseHandle(fd->fd);
		delete fd;
	}
}

int writeFile(FileHandle* fd, const  void* data,int32_t size) {
	if (fd) {
		DWORD dwWrited = 0;
		if (::WriteFile(fd->fd, data, size, &dwWrited, NULL)) {
			return (int)dwWrited;
		}
	}
	return 0;
}
int readFile(FileHandle* fd, uint8_t* buf, int32_t size) {
	if (fd) {
		DWORD dwReaded = 0;
		if (::ReadFile(fd->fd, buf, size, &dwReaded, NULL)) {
			return (int)dwReaded;
		}
	}
	return 0;
}

size_t getFileSize(FileHandle* fd) {
	return ::GetFileSize(fd->fd,NULL);
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
    if(mode==MODE_OPEN){	
		fd = ::open(path, O_RDWR | O_CREAT);
    }else if (mode == MODE_APPEND) {
		fd = ::open(path, O_RDWR | O_APPEND);
	} else {	  
        fd= ::open(path,O_RDWR | O_CREAT | O_TRUNC,777);
    }
	if (fd < 0) {
		return NULL;
	}
	FileHandle* handle= new FileHandle();
	handle->fd = fd;
    return handle;
}
void closeFile(FileHandle* fd) {
    close(fd->fd);
}

int writeFile(FileHandle* fd,void* data,int32_t size) {
    if (fd && fd->fd>=0) {
       int ret = (int)write(fd->fd, data, size);
       if (ret < 0){}

       return ret;
    }
    return 0;
}
int readFile(FileHandle* fd, uint8_t* buf, int32_t size) {
    if (fd  && fd->fd >= 0) {
       int ret = (int)read(fd->fd, buf, size);
       if (ret < 0){}
       return ret;
    }
    return 0;
}

size_t getFileSize(FileHandle* fd){
	FFL_ASSERT(0);
	return 0;
}
#endif

namespace FFL {
	File::File() {
		mFd = NULL;
	}
	File::~File() {
		close();		
	}
	status_t File::open(const String& path){		
		if (isOpened()) {
			return FFL_FILE_ALREADY_OPENED;			
		}

		this->open(path.string(), MODE_OPEN);
		mPath = path;
		return FFL_OK;
	}
	//
	//  追加模式打开文件，FFL_OK成功
	//  path:文件绝对路径
	//
	status_t File::openAppend(const String& path) {
		if (isOpened()) {
			return FFL_FILE_ALREADY_OPENED;
		}

		this->open(path.string(), MODE_APPEND);
		mPath = path;
		return FFL_OK;
	}
	//
	// 创建文件,文件已经存在的情况下覆盖原文件
	//
	status_t File::create(const String& path) {
		if (isOpened()) {
			return FFL_FILE_ALREADY_OPENED;
		}

		this->open(path.string(), MODE_ALWAYS_CREATE);
		mPath = path;
		return FFL_OK;
	}
	//
	//  打开文件，FFL_OK成功
	//  path:文件绝对路径
	//
	status_t File::open(const char* path, int mode) {
		mFd=createFile(path,(OpenFileMode)mode);
		return mFd != NULL ? FFL_OK:FFL_FILE_OPEN_FAILED;
	}

	void File::close(){
        if (mFd ==NULL) {
			return;
		}

		closeFile((FileHandle*)mFd);
		mFd = NULL;
	}

	bool File::isOpened() const{
		return mFd !=NULL;
	}
	//
	//  写数据到文件中
	//  buf:缓冲区地址
	//  count:缓冲区大小
	//  pWrite:实质上写了多少数据
	//  返回错误码  ： FFL_OK表示成功
	//
	status_t File::write(const void* buf, size_t count, size_t* pWrite){
		int ret = FFL_OK;

		int nWrited;
		if ((nWrited = writeFile((FileHandle*)mFd, (void*)buf, count)) < 0) {
			ret = FFL_FILE_WRITE_FAILED;
			return ret;
		}

		if (pWrite != NULL) {
			*pWrite = nWrited;
		}

		return ret;
	}

	status_t File::writeVec(const BufferVec* bufVec, int count, size_t* pWrite){
		int ret = FFL_OK;

		size_t nWrited = 0;
		for (int i = 0; i < count; i++) {
			const BufferVec* buf = bufVec + i;
			size_t n = 0;
			if ((ret = write(buf->data, buf->size, &n)) != FFL_OK) {
				return ret;
			}
			nWrited += n;
		}

		if (pWrite) {
			*pWrite = nWrited;
		}

		return ret;
	}

	//
	//  读数据到缓冲区
	//  buf:缓冲区地址
	//  count:需要读的大小
	//  pReaded:实质上读了多少数据
	//  返回错误码  ： FFL_OK表示成功
	//
	status_t File::read(uint8_t* buf, size_t count, size_t* pReaded) {
		int ret = FFL_OK;

		int nReaded;
		if ((nReaded = readFile((FileHandle*)mFd, buf, count)) < 0) {
			ret = FFL_FILE_READ_FAILED;
			return ret;
		}

		if (pReaded != NULL) {
			*pReaded = nReaded;
		}

		return ret;
	}

	//
	//  文件大小
	//
	size_t File::getSize() {
		if (mFd) {
			return getFileSize((FileHandle*)mFd);
		}

		return 0;			 
	}
	//
	//  文件是否创建了
	//
	bool fileIsExist(const char* path) {

#ifdef WIN32
		return  ::PathFileExistsA(path)?true:false;
#else		
		//
		//06     检查读写权限
		//04     检查读权限
		//02     检查写权限
		//01     检查执行权限
		//00     检查文件的存在性
		//
		return (access(path, 0) == 0);
#endif
	
	}
		
}


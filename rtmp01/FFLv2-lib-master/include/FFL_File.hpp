/* 
 *  This file is part of FFL project.
 *
 *  The MIT License (MIT)
 *  Copyright (C) 2017-2018 zhufeifei All rights reserved.
 *
 *  FFL_File.hpp
 *  Created by zhufeifei(34008081@qq.com) on 2018/06/20 
 *  https://github.com/zhenfei2016/FFLv2-lib.git
 *  文件操作类
 *
*/
#ifndef _FFL_FILE_HPP_
#define _FFL_FILE_HPP_

#include "FFL_Io.hpp"
#include "FFL_String.hpp"

namespace FFL {

	class FFLIB_API_IMPORT_EXPORT File : public IOReader ,public IOWriter {
	public:
		File();
		virtual ~File();
	public:
		//
		//  打开文件，FFL_OK成功
		//  path:文件绝对路径
		//
		status_t open(const String& path);
		//
		//  追加模式打开文件，FFL_OK成功
		//  path:文件绝对路径
		//
		status_t openAppend(const String& path);
		//
		// 创建文件,文件已经存在的情况下覆盖原文件
		//
		status_t create(const String& path);
		//
		// 关闭文件
		//
		void close();	
		bool isOpened() const;		
		//
		//  文件大小
		//
		size_t getSize();
	private:
		//
		//  打开文件，FFL_OK成功
		//  path:文件绝对路径
		//
		status_t open(const char* path,int mode);
	public:
		//
		//  写数据到文件中
		//  buf:缓冲区地址
		//  count:缓冲区大小
		//  pWrite:实质上写了多少数据
		//  返回错误码  ： FFL_OK表示成功
		//
		virtual status_t write(const void* buf, size_t count, size_t* pWrite);
		//
		//  写数据到文件中
		//  bufVec:缓冲区地址,数组
		//  count:数组大小
		//  pWrite:实质上写了多少数据
		//  返回错误码  ： FFL_OK表示成功
		//
		virtual status_t writeVec(const BufferVec* bufVec, int count, size_t* pWrite);
		//
		//  读数据到缓冲区
		//  buf:缓冲区地址
		//  count:需要读的大小
		//  pReaded:实质上读了多少数据
		//  返回错误码  ： FFL_OK表示成功
		//
		virtual status_t read(uint8_t* buf, size_t count, size_t* pReaded);
	private:
		//
		//  文件路径，文件句柄
		//
		String mPath;	
		void* mFd;
	};

	//
	//  文件是否创建了
	//
	bool fileIsExist(const char* path);
}

#endif

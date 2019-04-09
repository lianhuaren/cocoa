/*
*  This file is part of FFL project.
*
*  The MIT License (MIT)
*  Copyright (C) 2017-2018 zhufeifei All rights reserved.
*
*  FFL_Io.hpp
*  Created by zhufeifei(34008081@qq.com) on 2018/06/20
*  https://github.com/zhenfei2016/FFLv2-lib.git
*  io读写接口
*
*/

#ifndef _FFL_IO_HPP_
#define _FFL_IO_HPP_

#include <FFL_Core.h>
namespace FFL {
	//
	//  缓冲区buffer
	//
	struct BufferVec {
		void  *data;
		size_t size;
	};

	class IOReader {
	public:
		//
		//  读数据到缓冲区
		//  buf:缓冲区地址
		//  count:需要读的大小
		//  pReaded:实质上读了多少数据
		//  返回错误码  ： FFL_OK表示成功
		//
		virtual status_t read(uint8_t* buf, size_t count, size_t* pReaded)=0;
	};

	class IOWriter {
	public:
		//
		//  写数据到文件中
		//  buf:缓冲区地址
		//  count:缓冲区大小
		//  pWrite:实质上写了多少数据
		//  返回错误码  ： FFL_OK表示成功
		//
		virtual status_t write(const void* buf, size_t count, size_t* pWrite)=0;
		//
		//  写数据到文件中
		//  bufVec:缓冲区地址,数组
		//  count:数组大小
		//  pWrite:实质上写了多少数据
		//  返回错误码  ： FFL_OK表示成功
		//
		virtual status_t writeVec(const BufferVec* bufVec, int count, size_t* pWrite)=0;		
	};
}

#endif

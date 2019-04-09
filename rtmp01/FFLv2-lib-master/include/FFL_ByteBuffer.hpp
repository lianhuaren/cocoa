/*
*  This file is part of FFL project.
*
*  The MIT License (MIT)
*  Copyright (C) 2017-2018 zhufeifei All rights reserved.
*
*  FFL_ByteBuffer   
*  Created by zhufeifei(34008081@qq.com) on 2018/05/06 
*  https://github.com/zhenfei2016/FFLv2-lib.git
*  内存管理类
*
*/
#ifndef _FFL_BYTE_BUFFER_HPP_
#define _FFL_BYTE_BUFFER_HPP_

#include <FFL_Core.h>

namespace FFL{
	class ByteStream;
	class FFLIB_API_IMPORT_EXPORT ByteBuffer {
	public:
		ByteBuffer();
		ByteBuffer(uint32_t size);
		ByteBuffer(const uint8_t* data,uint32_t size);
		~ByteBuffer();
		
		//
		// 重新申请一下空间
		//
		uint32_t alloc(uint32_t size);
		//
		//  扩大一下内存空间,如果size小于已经申请的则返回以前的大小
		//
		uint32_t realloc(uint32_t size);
		//
		//  buffer指针
		//
		uint8_t* data() const;
		//
		//  buffer大小
		//
		uint32_t size() const;			
	private:
		uint8_t* mData;
		uint32_t mSize;
	};
}


#endif


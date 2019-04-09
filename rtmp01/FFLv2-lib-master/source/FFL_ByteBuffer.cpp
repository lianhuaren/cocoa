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

#include <FFL_ByteBuffer.hpp>
#include <FFL_ByteStream.hpp>
#include "internalLogConfig.h"

namespace FFL {
	static const int32_t kDefaultBufferSize = 4096;
	ByteBuffer::ByteBuffer():mData(0),mSize(0){		
		alloc(kDefaultBufferSize);
	}
	ByteBuffer::ByteBuffer(uint32_t size) : mData(0), mSize(0) {		
		alloc(size);
	}
	ByteBuffer::ByteBuffer(const uint8_t* data, uint32_t size) : mData(0), mSize(0) {		
		alloc(size);
		if (data && size) {
			memcpy(mData, data, size);
		}
	}
	ByteBuffer::~ByteBuffer(){
		FFL_free(mData);		
		mSize = 0;	
	}
	//
	// 重新申请一下空间
	//
	uint32_t ByteBuffer::alloc(uint32_t size) {
		if (size <= mSize) {
			return mSize;
		}
		if (size == 0) {
			size = 16;
		}

		uint8_t* data = (uint8_t*)FFL_malloc(size);
		if (!data) {
			INTERNAL_FFL_LOG_ERROR("ByteBuffer::alloc fail");
			return mSize;
		}	
		memset(data, 0, size);

		FFL_free(mData);
		mData = data;
		mSize = size;
		return size;
	}
	//
	//  扩大一下内存空间,如果size小于已经申请的则返回以前的大小
	//
	uint32_t ByteBuffer::realloc(uint32_t size) {
		if (size <= mSize) {
			return mSize;
		}

		uint8_t* data=(uint8_t*)FFL_malloc(size);
		if(!data){
			INTERNAL_FFL_LOG_ERROR("ByteBuffer::realloc fail");
			return mSize;
		}

		if (mSize) {
			memcpy(data, mData, mSize);
		}
		memset(data+ mSize, 0, size-mSize);

		FFL_free(mData);
		mData = data;
		mSize = size;		
		return size;
	}

	uint8_t* ByteBuffer::data() const {
		return mData;
	}
	uint32_t ByteBuffer::size() const {
		return mSize;
	}	
	
}

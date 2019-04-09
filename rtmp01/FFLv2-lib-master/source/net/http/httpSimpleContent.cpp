/*
*  This file is part of FFL project.
*
*  The MIT License (MIT)
*  Copyright (C) 2017-2018 zhufeifei All rights reserved.
*
*  httpSimpleContent.hpp
*  Created by zhufeifei(34008081@qq.com) on 2019/02/19
*  https://github.com/zhenfei2016/FFLv2-lib.git
*
*  http请求应答的内容接口
*/

#include "httpSimpleContent.hpp"
#include <FFL_ByteStream.hpp>
namespace FFL {	

	HttpSimpleContent::HttpSimpleContent(const uint8_t* data,int32_t size) {
		mBuffer = new ByteBuffer(data,size);
		mStream = new ByteStream();
		mStream->setData(mBuffer->data(), mBuffer->size(), mBuffer->size());
	}

	HttpSimpleContent::~HttpSimpleContent() {
		FFL_SafeFree(mBuffer);
		FFL_SafeFree(mStream);
	}
	//
	//  获取内容大小
	//
	int32_t HttpSimpleContent::getSize() {
		return mStream->getSize();
	}
	//
	//  获取内容
	//
	int32_t HttpSimpleContent::read(uint8_t* data, int32_t requestSize, bool* suc) {
		int32_t size = mStream->getSize();
		if (size >requestSize) {
			size = requestSize;
		} 		

		if (mStream->readBytes((int8_t*)data, size)) {
			if (suc) {
				*suc = true;
			}
		}
		else {
			if (suc) {
				*suc = false;
			}
		}
		return size;
	}
	

}


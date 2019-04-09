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
#ifndef _HTTP_SIMPLE_CONTENT_HPP_
#define _HTTP_SIMPLE_CONTENT_HPP_

#include <net/http/FFL_HttpHeader.hpp>
#include <FFL_ByteBuffer.hpp>
namespace FFL {	
	class HttpSimpleContent : public HttpContent {
	public:
		HttpSimpleContent(const uint8_t* data, int32_t size);
		~HttpSimpleContent();
	public:
		//
		//  获取内容大小
		//
		int32_t getSize();
		//
		//  获取内容
		//
		int32_t read(uint8_t* data, int32_t requestSize, bool* suc);
	
	protected:
		ByteBuffer* mBuffer;
		ByteStream* mStream;
	};			
}
#endif 



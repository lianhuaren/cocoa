/*
*  This file is part of FFL project.
*
*  The MIT License (MIT)
*  Copyright (C) 2017-2018 zhufeifei All rights reserved.
*
*  FFL_HttpHeader.hpp
*  Created by zhufeifei(34008081@qq.com) on 2018/07/14
*  https://github.com/zhenfei2016/FFLv2-lib.git
*  http头信息
*
*/

#ifndef _FFL_HTTp_HEADER_HPP_
#define _FFL_HTTp_HEADER_HPP_


#include <FFL_Core.h>
#include <FFL_Dictionary.hpp>
#include <FFL_RefBase.hpp>

namespace FFL {	
	//
	//  http头中的内容长度
	//
    #define HTTP_KEY_CONTENTLEN  "Content-Length"
	//
	//  http头中的内容类型
	//
    #define HTTP_KEY_CONTENTYPE  "Content-Type"

	class FFLIB_API_IMPORT_EXPORT HttpHeader : public Dictionary {
	public:
		HttpHeader();
		~HttpHeader();
		//
		// 获取content类型
		//
		void getContentType(String& type) ;
		void setContentType(const String& type);
		//
		// 获取content的长度
		//
		int32_t getContentLength() ;
		void setContentLength(int32_t len);
	protected:
		String  mContentType;
		int32_t mContentLength;		
	};

	//
	//  http发送的内容
	//
	class FFLIB_API_IMPORT_EXPORT  HttpContent : public RefBase {
	public:
		HttpContent();
		virtual ~HttpContent();

		//
		//  内容的长度
		//
		virtual int32_t getSize() = 0;
		//
		//  读取内容，依次向后读写的
		//  data:缓冲区
		//  requestSize:希望多少字节
		//  suc:返回是否成功了
		//  函数返回实质上返回的长短
		//  
		virtual int32_t read(uint8_t* data, int32_t requestSize,bool* suc)=0;
	};
	template class FFLIB_API_IMPORT_EXPORT FFL::sp<HttpContent>;
}

#endif

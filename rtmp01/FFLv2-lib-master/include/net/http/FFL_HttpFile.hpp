/*
*  This file is part of FFL project.
*
*  The MIT License (MIT)
*  Copyright (C) 2017-2018 zhufeifei All rights reserved.
*
*  FFL_HttpFile.hpp
*  Created by zhufeifei(34008081@qq.com) on 2019/02/19
*  https://github.com/zhenfei2016/FFLv2-lib.git
*
*  http文件请求，应答封装
*/
#ifndef _FFL_HTTP_FILE_HPP_
#define _FFL_HTTP_FILE_HPP_

#include <net/http/FFL_HttpResponse.hpp>
#include <net/http/FFL_HttpRequest.hpp>
#include <FFL_Io.hpp>

namespace FFL {	
	class HttpClient;
	class FFLIB_API_IMPORT_EXPORT HttpResponseFile : 
		public HttpResponse {
	public:
		HttpResponseFile(FFL::sp<HttpClient> client);			
		virtual ~HttpResponseFile();		

		//
		//  写内容到这个response中
		//
		bool writeFile(const char* filePath);
		//
		//  从response中读取文件
		//
		bool readFile(const char* filePath);
	};	


	class FFLIB_API_IMPORT_EXPORT HttpRequestFile :
		public HttpRequest {
	public:
		HttpRequestFile(FFL::sp<HttpClient> client);
		virtual ~HttpRequestFile();

		//
		//  写内容到这个response中
		//
		bool writeFile(const char* filePath);
		//
		//  从response中读取文件
		//
		bool readFile(const char* filePath);
	};
}

#endif
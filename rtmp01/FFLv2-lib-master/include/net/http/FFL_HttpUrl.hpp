/*
*  This file is part of FFL project.
*
*  The MIT License (MIT)
*  Copyright (C) 2017-2018 zhufeifei All rights reserved.
*
*  FFL_HttpUrl.hpp   
*  Created by zhufeifei(34008081@qq.com) on 2018/07/18
*  https://github.com/zhenfei2016/FFL-v2.git
*
*  分解URL
*/
#ifndef _HTTP_HTTP_URL_HPP_
#define _HTTP_HTTP_URL_HPP_
#include <FFL_Core.h>
#include <FFL_String.hpp>

namespace FFL {		
	class FFLIB_API_IMPORT_EXPORT HttpUrl{
	public:
		HttpUrl();
		~HttpUrl();
	public:		
		//
		// 输入类型： http://www.xxx.com:8000/v1/api
		//            mSchema:mHost mPort mPath mQuery
		bool parse(const String &url);
	public:		
		int32_t mPort;
		String mSchema;
		String mHost;		
		String mPath;
		String mQuery;
		//List<String> mQueryParams;
		
	};
}

#endif
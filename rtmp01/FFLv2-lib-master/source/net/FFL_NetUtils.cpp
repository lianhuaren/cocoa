/*
*  This file is part of FFL project.
*
*  The MIT License (MIT)
*  Copyright (C) 2017-2018 zhufeifei All rights reserved.
*
*  FFL_NetUtils.cpp
*  Created by zhufeifei(34008081@qq.com) on 2018/07/14
*  https://github.com/zhenfei2016/FFLv2-lib.git
*  帮组函数
*
*/

#include <FFL_String.hpp>

namespace FFL {

	void FFL_parseHostport(const String& url, String& host, int16_t& port){
		int pos = url.find(":");
		if (pos >=0) {			
			host.append(url.string(), pos );
			port = ::atoi(url.string()+pos+1);
		}else {
			host = url;
		}
	}

	//
	//  是否一个ip, 是的话返回FFL_Ok
	//
	status_t FFL_isIp(const String& ip) {
		return !ip.isEmpty() ?FFL_OK:FFL_FAILED;
	}
}
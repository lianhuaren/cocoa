/*
*  This file is part of FFL project.
*
*  The MIT License (MIT)
*  Copyright (C) 2017-2018 zhufeifei All rights reserved.
*
*  FFL_Time   
*  Created by zhufeifei(34008081@qq.com) on 2018/06/10 
*  https://github.com/zhenfei2016/FFLv2-lib.git
*  时间获取相关的封装函数
*
*/
#ifndef _FFL_TIME_H_
#define _FFL_TIME_H_

#include <FFL_Config.h>
#include <FFL_Stdint.h>

#ifdef  __cplusplus
extern "C" {
#endif
	/*  
	* sleep()
	*/
	FFLIB_API_IMPORT_EXPORT void  FFL_sleep(int ms);
	
	/*
	 *  获取当前时间 us
	 * */
	FFLIB_API_IMPORT_EXPORT int64_t FFL_getNowUs();
	FFLIB_API_IMPORT_EXPORT int32_t FFL_getNowMs();

	/*
	 *  获取当前时间的字符串类型
	 */
	FFLIB_API_IMPORT_EXPORT void FFL_getNowString(char* s,int32_t bufSize);

	/*
	*  毫秒转化为string， s:需要保证足够存储转化出来的时间字符串
	*/
	FFLIB_API_IMPORT_EXPORT void FFL_usToString(int64_t timeUs, char* s, int32_t bufSize);
#ifdef  __cplusplus
}
#endif
#endif 

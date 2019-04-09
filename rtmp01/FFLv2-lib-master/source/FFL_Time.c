/*
*  This file is part of FFL project.
*
*  The MIT License (MIT)
*  Copyright (C) 2017-2018 zhufeifei All rights reserved.
*
*  FFL_Time.c
*  Created by zhufeifei(34008081@qq.com) on 2018/06/10
*  https://github.com/zhenfei2016/FFLv2-lib.git
*  时间获取相关的封装函数
*
*/

#include <FFL_Core.h>
#include <FFL_Time.h>
#include <FFL_String.h>
#include <stdio.h>
//
//  是否使用短类型的时间字符串
//  短类型day hour min second ms us
//
#define FFL_TIME_USE_SHORT_STRING 0
#if defined( WIN32)
#include <windows.h>
#include "window_time.c"
#else
#include "linux_time.c"
#endif

void FFL_sleep(int ms)
{	
#if WIN32
	Sleep(ms);
#elif defined(MACOSX) || defined(ANDROID)
	int err;
	struct timespec elapsed, tv;
	elapsed.tv_sec = ms / 1000;
	elapsed.tv_nsec = (ms % 1000) * 1000000;
	do
	{
		tv.tv_sec = elapsed.tv_sec;
		tv.tv_nsec = elapsed.tv_nsec;
		err = nanosleep(&tv, &elapsed);
	} while (err);
#else
	FFL_ASSERT(0);
#endif
}

int64_t FFL_getNowUs() {
	int64_t nowUs=0;
#if WIN32	
	nowUs = (int64_t)internalGetUs();
#else
	nowUs = internalGetUs();
#endif
	return nowUs;
}

int32_t FFL_getNowMs() {
	int32_t now=(int32_t)(FFL_getNowUs()/1000);
	return now;
}


void FFL_getNowString(char* s, int32_t bufSize) {
	internalGetTimeString(internalGetUs(),s,bufSize);
}


/*
*  毫米转化为string，
*/
void FFL_usToString(int64_t timeUs, char* s, int32_t bufSize) {
	internalGetTimeString(timeUs, s,bufSize);
}

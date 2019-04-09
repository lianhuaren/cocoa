/*
*  This file is part of FFL project.
*
*  The MIT License (MIT)
*  Copyright (C) 2017-2018 zhufeifei All rights reserved.
*
*  FFL_Error.c
*  Created by zhufeifei(34008081@qq.com) on 2017/07/12
*  https://github.com/zhenfei2016/FFLv2-lib.git
*
*  错误，返回返回值定义
*/

#include <FFL_Error.h>
#include <FFL_Core.h>

/* 需要放到tls中 */
static int s_errorNO=0;
static char s_errorString[512];

/*设置获取错误*/
int FFL_set_error(const char* format,...)
{
	va_list args;
	va_start(args, format);	
	vsnprintf(s_errorString, 512 - 1, format, args);
	va_end(args);

	return FFL_ERROR_FAIL;
}
const char* FFL_get_error()
{
	return s_errorString;
}

int FFL_set_error_no(int errorno, const char* s)
{
	s_errorNO = errorno;
	FFL_set_error(s);
	return FFL_ERROR_FAIL;
}
const int FFL_get_error_no()
{
	return s_errorNO;
}


int FFL_outofmemory()
{
	FFL_ASSERT(0);	
	return FFL_ERROR_SUCCESS;
}

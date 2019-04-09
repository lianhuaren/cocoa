/*
*  This file is part of FFL project.
*
*  The MIT License (MIT)
*  Copyright (C) 2017-2018 zhufeifei All rights reserved.
*
*  FFL_String.c
*  Created by zhufeifei(34008081@qq.com) on 2017/7/12.
*  https://github.com/zhenfei2016/FFLv2-lib.git
*
*  字符串操作函数
*/

#include <FFL_String.h>
#include <FFL_Memory.h>

char* FFL_strdup(const char *s)
{
	char *ptr = NULL;
	if (s) {
		size_t len = strlen(s) + 1;
		ptr = FFL_mallocz(len);
		if (ptr)
			memcpy(ptr, s, len);
	}
	return ptr;
}

char* FFL_strndup(const char *s, size_t len)
{
	char *ret = NULL, *end;
	if (!s)
		return NULL;

	end = memchr(s, 0, len);
	if (end)
		len = end - s;

	ret = FFL_mallocz(len + 1);
	if (!ret)
		return NULL;

	memcpy(ret, s, len);
	ret[len] = 0;
	return ret;
}

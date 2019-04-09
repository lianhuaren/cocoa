/*
*  This file is part of FFL project.
*
*  The MIT License (MIT)
*  Copyright (C) 2017-2018 zhufeifei All rights reserved.
*
*  FFL_String   
*  Created by zhufeifei(34008081@qq.com) on 2017/7/12. 
*  https://github.com/zhenfei2016/FFLv2-lib.git
*
*  内存的申请，释放，大小端控制 ， 内存泄漏检测
*/

#ifndef _FFL_STRING_H_
#define _FFL_STRING_H_

#include <stdlib.h>
#include <memory.h>
#include <FFL_Config.h>
#include <FFL_Stdint.h>
#include <string.h>

#ifdef  __cplusplus
extern "C" {
#endif
/*  
*  字符串复制
*/
FFLIB_API_IMPORT_EXPORT char* FFL_strdup(const char *s);
FFLIB_API_IMPORT_EXPORT char* FFL_strndup(const char *s, size_t len);
	

/*  win平台在1600（vs2010）版本前需要自己定义snprintf  */
#if WIN32
#if _MSC_VER <=1600
  #define snprintf _snprintf
#elif FFLIB_COMPILE_SHARED
  #pragma comment(lib,"legacy_stdio_definitions.lib")
#endif

#endif


#ifdef  __cplusplus
}
#endif
#endif

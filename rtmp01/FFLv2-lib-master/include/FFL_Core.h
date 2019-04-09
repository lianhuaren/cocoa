/*
*  This file is part of FFL project.
*
*  The MIT License (MIT)
*  Copyright (C) 2017-2018 zhufeifei All rights reserved.
*
*  FFL_Core   
*  Created by zhufeifei(34008081@qq.com) on 2017/07/10 
*  https://github.com/zhenfei2016/FFLv2-lib.git
*
*
*/
#ifndef _FFL_CORE_H_
#define _FFL_CORE_H_

/*  字符串操作 */
#include <string.h>
#include <FFL_Config.h>
#include <FFL_Platform.h>
#include <FFL_Stdint.h>
#include <FFL_Error.h>
#include <FFL_Assert.h>
#include <FFL_Memory.h>
#include <FFL_String.h>
#include <FFL_Time.h>
#include <FFL_Log.h>
#include <FFL_LogConfig.h>

/*
*  函数调用方式
*/
#ifndef FFL_CALL
#if (defined(WIN32))
#define FFL_CALL __cdecl
#elif (defined(MACOSX))
#define FFL_CALL
#else
#define FFL_CALL
#endif
#endif

/*
* 内联函数
*/
#define FFL_INLINE 


/*  最大值 */
#ifndef FFL_MAX
#define FFL_MAX(a, b)    ((a) > (b) ? (a) : (b))
#endif

/*  最小值 */
#ifndef FFL_MIN
#define FFL_MIN(a, b)    ((a) < (b) ? (a) : (b))
#endif

/*  align字节对齐 */
#ifndef FFL_ALIGN
#define FFL_ALIGN(x, align) ((( x ) + (align) - 1) / (align) * (align))
#endif

/* 数组元素个数 */
#ifndef FFL_ARRAY_ELEMS
#define FFL_ARRAY_ELEMS(x) ((int) (sizeof(x) / sizeof((x)[0])))
#endif

/* 交换2个数值 */
#ifndef FFL_SWAP
#define FFL_SWAP(type,a,b) do{type tmp= b; b= a; a= tmp;}while(0)
#endif

/*
*  转换到字符串类型
*/
#ifndef FFL_TOSTRING
#define FFL_TOSTRING(s) #s
#endif

#ifndef FFL_INT32_HIGHT_16
#define FFL_INT32_HIGHT_16(a) ((((int)(a))>>16) & 0xFFFF)
#endif

#ifndef FFL_INT32_LOW_16
#define FFL_INT32_LOW_16(a) ((((int)(a))) & 0xFFFF)
#endif

#ifndef FFL_MAKE_INT32
#define FFL_MAKE_INT32(high,low) (((low)& 0xFFFF) | (((high)& 0xFFFF)<<16))
#endif


#ifndef FFL_INT64_HIGHT_32
#define FFL_INT64_HIGHT_32(a) ((int32_t)( ((a) >> 32) & 0xFFFFFFFF))
#endif

#ifndef FFL_INT64_LOW_32
#define FFL_INT64_LOW_32(a) ((int32_t)( (a) & 0xFFFFFFFF ))
#endif

#ifndef FFL_MAKE_INT64
#define FFL_MAKE_INT64(high,low) (((int64_t)(low)& 0xFFFFFFFF) | (((int64_t)(high)& 0xFFFFFFFF)<<32))
#endif

/*
*打印64位整形
* print("hi%"lld64, 123445 );
*/
#if defined(WIN32)
#define lld64 "I64d"
#else
#define lld64 "lld"
#endif

/*
*  socket句柄
*/
typedef int NetFD;
#define INVALID_NetFD 0

#ifdef  __cplusplus
extern "C"{
#endif
/*
*  初始化函数
*
* */
FFLIB_API_IMPORT_EXPORT void FFL_CALL FFL_initialize();

/*
*  结束FFL库的使用，调用后就不要再使用FFL的函数了
*
* */
FFLIB_API_IMPORT_EXPORT void FFL_CALL FFL_terminate();

#ifdef  __cplusplus
}
#endif

#endif

#ifdef  __cplusplus
#ifndef _FFL_CORE_HPP_
#define _FFL_CORE_HPP_
#include <FFL_Utils.hpp>
#endif
#endif
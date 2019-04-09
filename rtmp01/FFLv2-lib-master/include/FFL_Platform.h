/*
*  This file is part of FFL project.
*
*  The MIT License (MIT)
*  Copyright (C) 2017-2018 zhufeifei All rights reserved.
*
*  FFL_Platform   
*  Created by zhufeifei(34008081@qq.com) on 2017/07/12
*  https://github.com/zhenfei2016/FFLv2-lib.git
*  根据编译器定义的编译变量，确定编译的目标平台
*
*/

#ifndef _FFL_PLATFORM_H_
#define _FFL_PLATFORM_H_

#ifdef __cplusplus
extern "C" {
#endif

/*  
 * win32 平台
 */
#ifndef WIN32
#if defined( _MSC_VER) || defined(_WIN32) || defined(__WIN32__) 
#define WIN32 
#endif 
#endif

/* 
 * macos平台 
 */
#ifndef MACOSX
#if defined(__MACOSX__) || defined(__APPLE__)
#define MACOSX 1
#endif 
#endif

/* 
 *apple  ios平台 
 */
#ifndef IOS
#if defined(__APPLE__)
#define IOS 1
#endif
#endif

/*
 *android 平台下 
 */
#ifndef ANDROID
#if defined(__ANDROID__)
#define ANDROID
#endif
#endif

/*
*  打印一下在那个平台下编译的
*
*/
#define FFL_COMPILER_LOG(f) 
#if defined(WIN32)
	FFL_COMPILER_LOG(target platform win32)
#elif defined(IOS)
	FFL_COMPILER_LOG(target platform ios)
#elif defined(MACOSX)
	FFL_COMPILER_LOG(target platform macosx)
#elif defined(ANDROID)
	FFL_COMPILER_LOG(target platform android)
#else
	FFL_COMPILER_LOG(target platform not support)
#endif

#ifdef __cplusplus
}
#endif

#endif

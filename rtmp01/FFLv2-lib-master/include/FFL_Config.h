/*
*  This file is part of FFL project.
*
*  The MIT License (MIT)
*  Copyright (C) 2017-2018 zhufeifei All rights reserved.
*
*  FFL_Config   
*  Created by zhufeifei(34008081@qq.com) on 2017/07/10 
*  https://github.com/zhenfei2016/FFLv2-lib.git
*
*  编译前生成的配置文件 ,不依赖其他文件的，
*/

#ifndef _FFL_CONFIG_H_
#define _FFL_CONFIG_H_


#define FFLIB_API_IMPORT 
#define FFLIB_API_EXPORT 
//
//  如果脚本定义了编译动态库
#if  defined(FFL_EXPORTS)
#define FFLIB_API_IMPORT_EXPORT FFLIB_API_EXPORT
#else
#define FFLIB_API_IMPORT_EXPORT FFLIB_API_IMPORT
#endif


/*
 *  是否检测内存泄露
 */
#define CHECK_FOR_MEMORY_LEAKS 0

/*    线程相关    */
//#define FFL_THREAD_WINDOWS 0
#define FFL_THREAD_WINDOWS 0

//#define FFL_THREAD_STDCPP 0
#define FFL_THREAD_STDCPP 0

//#define FFL_THREAD_PTHREAD 0
#define FFL_THREAD_PTHREAD 1

#if FFL_THREAD_PTHREAD
#define HAVE_PTHREADS  1
#endif

#endif

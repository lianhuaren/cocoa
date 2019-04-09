/*
*  This file is part of FFL project.
*
*  The MIT License (MIT)
*  Copyright (C) 2017-2018 zhufeifei All rights reserved.
*
*  FFL_Memory   
*  Created by zhufeifei(34008081@qq.com) on 2017/7/12. 
*  https://github.com/zhenfei2016/FFLv2-lib.git
*
*  内存的申请，释放，大小端控制 ， 内存泄漏检测
*/

#ifndef _FFL_MEMORY_H_
#define _FFL_MEMORY_H_

#include <stdlib.h>
#include <memory.h>
#include <FFL_Config.h>
#include <FFL_Stdint.h>

#ifdef  __cplusplus
extern "C" {
#endif
	/*
	 *  内存申请，释放
	 */
	FFLIB_API_IMPORT_EXPORT void* FFL_malloc(size_t size);
	FFLIB_API_IMPORT_EXPORT void  FFL_free(void *mem);
	/*
	* 申请内存并且清零
	*/
	FFLIB_API_IMPORT_EXPORT void* FFL_mallocz(size_t size);
	
#define FFL_freep(pp)  while(pp&&*pp){ FFL_free(*pp); *pp = 0; break;}
	/*
	*   memory清零
	*/
#define FFL_Zerop(x) do{memset((x), 0, sizeof(*(x)));} while(0)
#define FFL_ZeroArray(x,c) do{memset((x), 0, (sizeof(*(x))*(c)));} while(0)
	
	/*
	*  内存字节序
	*大端模式，数据的低位保存在内存的高地址中
	* 网络字节序
	*/
#define  FFL_BIG_ENDIAN 1
	/*
	*小端模式，数据的低位保存在内存的低地址中
	*/
#define  FFL_LITTLE_ENDIAN 2
	/*
	*  检测系统的大小端
	*/
	FFLIB_API_IMPORT_EXPORT int FFL_isLittleEndian();

	/*
	   order: 1:顺序的，0:还是反序的  拷贝数据
	*/
	FFLIB_API_IMPORT_EXPORT void FFL_copyBytes(uint8_t* s, uint8_t* d, uint32_t size, int order);

	/************************memoryleak相关*********************************************************************/
	/*
	 *   打印一下当前还没有释放的内存
	 */
	FFLIB_API_IMPORT_EXPORT void  FFL_dumpMemoryLeak();
	/*
	*  打印当前未释放的内存，到文件中
	*/
	FFLIB_API_IMPORT_EXPORT void  FFL_dumpMemoryLeakFile(const char* path);
	/*
 	 *  参考上一次释放的内存文件，打印对应的堆栈
	 */
	FFLIB_API_IMPORT_EXPORT void  FFL_checkMemoryLeak(const char* path);	
#ifdef  __cplusplus
}
#endif
#endif

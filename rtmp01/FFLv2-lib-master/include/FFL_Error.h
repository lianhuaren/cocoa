/*
*  This file is part of FFL project.
*
*  The MIT License (MIT)
*  Copyright (C) 2017-2018 zhufeifei All rights reserved.
*
*  FFL_Error   
*  Created by zhufeifei(34008081@qq.com) on 2017/07/12 
*  https://github.com/zhenfei2016/FFLv2-lib.git
*
*  错误，返回返回值定义
*/
#ifndef _FFL_ERROR_H_
#define _FFL_ERROR_H_

#include <FFL_Config.h>
#include <FFL_Stdint.h>

typedef enum  ERROR_NO
{
	/*  返回成功  */
	FFL_OK = 0,
	FFL_NO_ERROR = FFL_OK,
	FFL_ERROR_SUCCESS = FFL_OK,

	/*  无效参数  */
	ERROR_INVALID_PARAMS = -1,

	/*  发生错误了  */
	FFL_ERROR_FAIL = -1,
	FFL_ERROR_FAILED = -1,
	FFL_FAILED = -1,

	/*
	 * 无效操作
	 */
	FFL_INVALID_OPERATION = -2,
	FFL_INVALID_PARAMS = -3,

	/*
	 * 没实现
	 */
	FFL_NOT_IMPLEMENT = -4,

	/*
	 *  函数执行错误，不能block
	 */
	FFL_WOULD_BLOCK = -5,

	/*
	* 未初始化
	*/
	FFL_NOT_INITIALIZE = -6,

	FFL_ERROR_EOF = -200,

	/*  超时  */
	ERROR_TIME_OUT = -100,


	FFL_MUTEX_TIMEDOUT = -100,
	FFL_MUTEX_MAXWAIT = ~0,


	/*  pipeline   已经填充了一个buffer */
	PIPELINE_FILL_BUFFER = 1000,

	/*
	 *  跳过这个buffer
	 *  */
	PIPELINE_SKIP_BUFFER,

	/*
	 *  文件已经打开了
	 */
	FFL_FILE_ALREADY_OPENED,
	/*
	*  文件打开失败了
	*/
	FFL_FILE_OPEN_FAILED,
	FFL_FILE_CLOSE_FAILED,
	FFL_FILE_WRITE_FAILED,
	FFL_FILE_READ_FAILED,

	

}ERROR_NO;

/*
 * 函数返回值，表示函数的执行状态
*/
typedef int status_t;

#ifdef  __cplusplus
extern "C" {
#endif
	/* 设置获取错误 */
	int FFL_set_error(const char* format, ...);
	const char* FFL_get_error();

	int FFL_set_error_no(int errorno, const char* s);
	const int FFL_get_error_no();


	FFLIB_API_IMPORT_EXPORT int FFL_outofmemory();

#ifdef  __cplusplus
}
#endif

#endif

/*
*  This file is part of FFL project.
*
*  The MIT License (MIT)
*  Copyright (C) 2017-2018 zhufeifei All rights reserved.
*
*  FFL_Atomic.h   
*  Created by zhufeifei(34008081@qq.com) on 2017/7/12.
*  https://github.com/zhenfei2016/FFLv2-lib.git
*
*/

#ifndef _FFL_ATOMIC_H_
#define _FFL_ATOMIC_H_

#include <FFL_Core.h>

#ifdef WIN32

#else
#define FFL_ATOMIC_STDCPP
#endif

#ifdef FFL_ATOMIC_STDCPP
typedef void* AtomicInt;
#else
typedef volatile long AtomicInt;
#endif 

#ifdef __cplusplus
extern "C" {
#endif 
	/*
	*  初始化 atomicVal 值为 initValue，
	*/
	void FFL_atomicInit(AtomicInt* atomicVal, int initValue);

	/*
	*  反初始化 atomicVal，调用以后这个值就不要再使用了
	*/
	void FFL_atomicUninit(AtomicInt* atomicVal);

	/*
	*	自增
	*/
	int FFL_atomicInc(AtomicInt* dest);
	/*
	*	自减
	*/
	int FFL_atomicDec(AtomicInt* dest);
	/*
	* 增加那么多
	*/
	int FFL_atomicAdd(AtomicInt* dest, int add);

	/*
	* cmp与*dest比较如果相同则把exchange赋值给dest
	* 返回0表示成功
	*/
	int FFL_atomicCmpxchg(AtomicInt *dest, int cmp, int exchange);

	/*
	* 设置新的数值
	*/
	void FFL_atomicSet(AtomicInt* atomicVal, int value);

	/*
	*   比较 atomicval更一个整形的值 ,
	*   返回0：表示不成功
	*       1: 成功
	*/
	int FFL_atomicValueEqual(AtomicInt* atomicVal, int value);

	/*
	* 获取这个atomic变量的整形值
	*/
	int FFL_atomicValueGet(const AtomicInt* atomicVal);

	/*
	* 获取这个atomic变量与 value 的and值的整形值
	* (*atomicVal & value)
	*/
	int FFL_atomicValueAnd(const AtomicInt* atomicVal, int value);

#ifdef __cplusplus
}
#endif 

#endif 

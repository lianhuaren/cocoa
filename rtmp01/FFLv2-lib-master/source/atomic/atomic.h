/*
*  This file is part of FFL project.
*
*  The MIT License (MIT)
*  Copyright (C) 2017-2018 zhufeifei All rights reserved.
*
*  atomic   
*  Created by zhufeifei(34008081@qq.com) on 2018/08/02 
*  https://github.com/zhenfei2016/FFLv2-lib.git
*  atomic系列函数声明
*/
#ifndef _INTERNAL_ATOMIC_H_
#define _INTERNAL_ATOMIC_H_

#include <FFL_Atomic.h>

#ifdef __cplusplus
extern "C" {
#endif 

	/*
	 *  初始化 atomicVal 值为 initValue，
	 */
	void atomicInit(AtomicInt* atomicVal, int initValue);

	/*
	 *  反初始化 atomicVal，调用以后这个值就不要再使用了
	 */
	void atomicUninit(AtomicInt* atomicVal);

	/*
	 *	自增
	 */
	int atomicInc(AtomicInt* dest);
	/*
	 *	自减
	 */
	int atomicDec(AtomicInt* dest);
	/*
	 * 增加那么多
	 */
	int atomicAdd(AtomicInt* dest, int add);

	/*
	 * cmp与*dest比较如果相同则把exchange赋值给dest
	 * 返回0表示成功
	*/
	int atomicCmpxchg(AtomicInt *dest, int cmp, int exchange);

	/*
	 * 设置新的数值
	 */
	void atomicSet(AtomicInt* atomicVal, int value);

	/*
	 *   比较 atomicval更一个整形的值 ,
	 *   返回0：表示不成功
	 *       1: 成功
	 */
	int atomicValueEqual(AtomicInt* atomicVal, int value);

	/*
	 * 获取这个atomic变量的整形值
	 */
	int atomicValueGet(const AtomicInt* atomicVal);

	/*
	 * 获取这个atomic变量与 value 的and值的整形值
	 * (*atomicVal & value)
	 */
	int atomicValueAnd(const AtomicInt* atomicVal, int value);

#ifdef __cplusplus
}
#endif 

#endif

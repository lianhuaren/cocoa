#ifndef FFL_ATOMIC_WINDOW_HPP_
#define FFL_ATOMIC_WINDOW_HPP_
#include <windows.h>
//#include "atomic.h"


extern "C" void atomicInit(AtomicInt* atomicVal, int initValue)
{
	*atomicVal = initValue;
}

/*
*  反初始化 atomicVal，调用以后这个值就不要再使用了
*/
extern "C" void atomicUninit(AtomicInt* atomicVal)
{	
}

/*自增,返回原始值 */
extern "C" int atomicInc(AtomicInt* dest)
{
	return atomicAdd(dest, 1);
}
/*
*自减
*/
extern "C" int atomicDec(AtomicInt* dest)
{
	return atomicAdd(dest ,-1);
}

/*
 *  增加add,返回原始值
 */
extern "C" int atomicAdd(AtomicInt* dest, int add)
{
	return InterlockedExchangeAdd(dest,add);	
}

/*
 *cmp与*atomicVal比较如果不相同则把exchange赋值给dest
 *返回0表示成功
*/
extern "C" int atomicCmpxchg(AtomicInt *atomicVal, int cmp, int exchange)
{
	if (InterlockedCompareExchange(atomicVal, exchange,cmp)== cmp)
	{
		return 0;
	}	
	return 1;	
}

extern "C" void atomicSet(AtomicInt* dest, int value)
{
	InterlockedExchange(dest, value);
}

extern "C" int atomicValueEqual(AtomicInt* atomicVal, int value)
{
	return (InterlockedExchangeAdd(atomicVal,0)== value)?1:0;
}

extern "C" int atomicValueGet(const AtomicInt* atomicVal) {
	return *atomicVal;
}


/*
* 获取这个atomic变量与 value 的and值的整形值
* (*atomicVal & value)
*/
int atomicValueAnd(const AtomicInt* atomicVal, int value) {
	int a = atomicValueGet(atomicVal);
	return a&value;
}


#endif

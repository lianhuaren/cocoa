#ifndef FFL_ATOMIC_STDCPLUS11_HPP_
#define FFL_ATOMIC_STDCPLUS11_HPP_

#include <atomic>

static std::atomic<int>& internalGetRealType(const AtomicInt* atomicVal)
{
	return *((std::atomic<int>*)(*atomicVal));
}

/*
*  初始化 atomicVal 值为 initValue，
*/
extern "C" void atomicInit(AtomicInt* atomicVal, int initValue)
{
	*atomicVal = new std::atomic<int>(initValue);
}

/*
*  反初始化 atomicVal，调用以后这个值就不要再使用了
*/
extern "C" void atomicUninit(AtomicInt* atomicVal)
{
	delete (std::atomic<int>*)(*atomicVal);
	*atomicVal = 0;
}

/*自增,返回原始值 */
extern "C"  int atomicInc(AtomicInt* dest)
{
	return internalGetRealType(dest)++;
}
//自减
extern "C"  int atomicDec(AtomicInt* dest)
{
	return internalGetRealType(dest)--;
}

/*
增加那么多
*/
extern "C" int atomicAdd(AtomicInt* dest, int add)
{
	return internalGetRealType(dest)+=add;	
}

/*
 *cmp与*atomicVal比较如果不相同则把exchange赋值给dest
 *返回0表示成功
*/
extern "C" int atomicCmpxchg(AtomicInt *atomicVal, int cmp, int exchange)
{
	if (internalGetRealType(atomicVal).compare_exchange_strong(cmp,exchange))
	{
		return 0;
	}	
	return 1;	
}

extern "C" void atomicSet(AtomicInt* dest, int value)
{
	internalGetRealType(dest).store(value);
}

extern "C" int atomicValueEqual(AtomicInt* atomicVal, int value)
{
	return (internalGetRealType(atomicVal)== value)?1:0;
}

extern "C" int atomicValueGet(const AtomicInt* atomicVal) {
	return internalGetRealType(atomicVal).load();
}


/*
* 获取这个atomic变量与 value 的and值的整形值
* (*atomicVal & value)
*/
extern "C" int atomicValueAnd(const AtomicInt* atomicVal, int value) {
	int a = atomicValueGet(atomicVal);
	return a&value;
}


#endif

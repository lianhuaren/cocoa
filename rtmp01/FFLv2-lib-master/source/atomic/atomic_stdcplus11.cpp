#ifndef FFL_ATOMIC_STDCPLUS11_HPP_
#define FFL_ATOMIC_STDCPLUS11_HPP_

#include <atomic>

static std::atomic<int>& internalGetRealType(const AtomicInt* atomicVal)
{
	return *((std::atomic<int>*)(*atomicVal));
}

/*
*  ��ʼ�� atomicVal ֵΪ initValue��
*/
extern "C" void atomicInit(AtomicInt* atomicVal, int initValue)
{
	*atomicVal = new std::atomic<int>(initValue);
}

/*
*  ����ʼ�� atomicVal�������Ժ����ֵ�Ͳ�Ҫ��ʹ����
*/
extern "C" void atomicUninit(AtomicInt* atomicVal)
{
	delete (std::atomic<int>*)(*atomicVal);
	*atomicVal = 0;
}

/*����,����ԭʼֵ */
extern "C"  int atomicInc(AtomicInt* dest)
{
	return internalGetRealType(dest)++;
}
//�Լ�
extern "C"  int atomicDec(AtomicInt* dest)
{
	return internalGetRealType(dest)--;
}

/*
������ô��
*/
extern "C" int atomicAdd(AtomicInt* dest, int add)
{
	return internalGetRealType(dest)+=add;	
}

/*
 *cmp��*atomicVal�Ƚ��������ͬ���exchange��ֵ��dest
 *����0��ʾ�ɹ�
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
* ��ȡ���atomic������ value ��andֵ������ֵ
* (*atomicVal & value)
*/
extern "C" int atomicValueAnd(const AtomicInt* atomicVal, int value) {
	int a = atomicValueGet(atomicVal);
	return a&value;
}


#endif

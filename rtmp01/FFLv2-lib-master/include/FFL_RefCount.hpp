/*
*  This file is part of FFL project.
*
*  The MIT License (MIT)
*  Copyright (C) 2017-2018 zhufeifei All rights reserved.
*
*  FFL_RefBase
*  Created by zhufeifei(34008081@qq.com) on 2017/08/12
*  https://github.com/zhenfei2016/FFLv2-lib.git
*   基于android中的引用计数类修改的
*
*/

#ifndef _FFL_REF_COUNT_H_
#define _FFL_REF_COUNT_H_

#include "FFL_RefAtomic.hpp"

namespace FFL
{
	template <class T>
	class Refcount
	{
	public:
		inline Refcount()  
		{ 
			FFL_atomicInit(&mCount,0);
		}

		inline void incStrong(const void* id) const 
		{
			FFL_atomicInc(&mCount);
		}
		inline void decStrong(const void* id) const
		{
			if (FFL_atomicDec(&mCount) == 1) {
				delete static_cast<const T*>(this);
			}
		}	
		inline int32_t getStrongCount() const
		{
			return (int32_t)mCount;
		}

		typedef Refcount<T> basetype;

	protected:
		inline ~Refcount() { }
	private:
		mutable AtomicInt mCount;
	};
};
#endif 

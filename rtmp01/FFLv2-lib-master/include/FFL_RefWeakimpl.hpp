/*
*  This file is part of FFL project.
*
*  The MIT License (MIT)
*  Copyright (C) 2017-2018 zhufeifei All rights reserved.
*
*  FFL_ref_weakimpl
*  Created by zhufeifei(34008081@qq.com) on 2017/08/12
*  https://github.com/zhenfei2016/FFLv2-lib.git
*
*  引用计数类，内部增加的引用计数相关结构
*/

#ifndef _FFL_REF_WEAK_REF_IMPL_H_
#define _FFL_REF_WEAK_REF_IMPL_H_

#include "FFL_RefBase.hpp"
#include "FFL_RefAtomic.hpp"

namespace FFL
{
    #define INITIAL_STRONG_VALUE (1<<28)
	class weakref_impl : public weakref_type
	{
	public:
		AtomicInt    mStrong;
		AtomicInt    mWeak;
		RefBase* const      mBase;
		AtomicInt    mFlags;

	public:
		
		weakref_impl(RefBase* base);
		~weakref_impl();

		void addStrongRef(const void* /*id*/);
		void removeStrongRef(const void* /*id*/);
		void renameStrongRefId(const void* /*old_id*/, const void* /*new_id*/);
		void addWeakRef(const void* /*id*/);
		void removeWeakRef(const void* /*id*/);
		void renameWeakRefId(const void* /*old_id*/, const void* /*new_id*/);
		void printRefs() const;
		void trackMe(bool, bool);
	};
};

#endif

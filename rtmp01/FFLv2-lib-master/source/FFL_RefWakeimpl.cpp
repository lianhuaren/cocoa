/*
*  This file is part of FFL project.
*
*  The MIT License (MIT)
*  Copyright (C) 2017-2018 zhufeifei All rights reserved.
*
*  FFL_ref_wakeimpl.cpp
*  Created by zhufeifei(34008081@qq.com) on 2017/08/12
*  https://github.com/zhenfei2016/FFLv2-lib.git

*

*/
#include "FFL_RefWeakimpl.hpp"

namespace FFL
{
	weakref_impl::weakref_impl(RefBase* base): mBase(base)
	{
		FFL_atomicInit(&mStrong,INITIAL_STRONG_VALUE);
		FFL_atomicInit(&mWeak, 0);
		//
		//  默认0强引用控制对象生命
		//
		FFL_atomicInit(&mFlags, 0);
	}

	weakref_impl::~weakref_impl() 
	{
		FFL_atomicUninit(&mStrong);
		FFL_atomicUninit(&mWeak);
		FFL_atomicUninit(&mFlags);
	}
	void weakref_impl::addStrongRef(const void* /*id*/) { }
	void weakref_impl::removeStrongRef(const void* /*id*/) { }
	void weakref_impl::renameStrongRefId(const void* /*old_id*/, const void* /*new_id*/) { }
	void weakref_impl::addWeakRef(const void* /*id*/) { }
	void weakref_impl::removeWeakRef(const void* /*id*/) { }
	void weakref_impl::renameWeakRefId(const void* /*old_id*/, const void* /*new_id*/) { }
	void weakref_impl::printRefs() const { }
	void weakref_impl::trackMe(bool, bool) { }



};

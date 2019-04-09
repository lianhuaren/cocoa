/*
*  This file is part of FFL project.
*
*  The MIT License (MIT)
*  Copyright (C) 2017-2018 zhufeifei All rights reserved.
*
*  FFL_Mutex
*  Created by zhufeifei(34008081@qq.com) on 2018/06/10
*  https://github.com/zhenfei2016/FFLv2-lib.git
*
*  锁，条件变量包装
*/

#include <FFL_Mutex.hpp>

namespace FFL {
	CMutex::CMutex() 
	{
		m_mutex = FFL_CreateMutex();
	}
	CMutex::~CMutex()
	{
		FFL_DestroyMutex(m_mutex);
		m_mutex = 0;
	}

	status_t    CMutex::lock()
	{
		return FFL_LockMutex(m_mutex);		
	}
	void        CMutex::unlock()
	{
		FFL_UnlockMutex(m_mutex);
	}
	status_t    CMutex::tryLock()
	{
		return FFL_TryLockMutex(m_mutex);
	}

	CCondition::CCondition()
	{
		m_cond = FFL_CreateCond();
	}
	CCondition::~CCondition()
	{
		FFL_DestroyCond(m_cond);
	}

	status_t CCondition::wait(CMutex& mutex)
	{
		return FFL_CondWait(m_cond, mutex.m_mutex);
	}
	status_t CCondition::waitRelative(CMutex& mutex, uint32_t ms)
	{
		return FFL_CondWaitTimeout(m_cond, mutex.m_mutex, ms);
	}

	void CCondition::signal()
	{
		FFL_CondSignal(m_cond);
	}
	void CCondition::broadcast()
	{
		FFL_CondBroadcast(m_cond);
	}
}

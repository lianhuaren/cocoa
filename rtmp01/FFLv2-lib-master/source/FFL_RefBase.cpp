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

#define LOG_TAG "RefBase"
// #define LOG_NDEBUG 0

#include "FFL_RefBase.hpp"
#include "FFL_RefLog.hpp"
#include "FFL_RefAtomic.hpp"
#include "FFL_RefWeakimpl.hpp"

// compile with refcounting debugging enabled
#define DEBUG_REFS                      0

// whether ref-tracking is enabled by default, if not, trackMe(true, false)
// needs to be called explicitly
#define DEBUG_REFS_ENABLED_BY_DEFAULT   0

// whether callstack are collected (significantly slows things down)
#define DEBUG_REFS_CALLSTACK_ENABLED    1

// folder where stack traces are saved when DEBUG_REFS is enabled
// this folder needs to exist and be writable
#define DEBUG_REFS_CALLSTACK_PATH       "/data/debug"

// log all reference counting operations
#define PRINT_REFS                      0

// ---------------------------------------------------------------------------


namespace FFL {
	RefBase::RefBase()
		: mRefs(new weakref_impl(this))
	{
	}

	RefBase::~RefBase()
	{
		if (FFL_atomicValueEqual(&(mRefs->mStrong),INITIAL_STRONG_VALUE)) {
			// we never acquired a strong (and/or weak) reference on this object.
			delete mRefs;
		}
		else {
			// life-time of this object is extended to WEAK or FOREVER, in
			// which case weakref_impl doesn't out-live the object and we
			// can free it now.
			if (FFL_atomicValueAnd(&(mRefs->mFlags), OBJECT_LIFETIME_MASK)
				 != OBJECT_LIFETIME_STRONG) {
				// It's possible that the weak count is not 0 if the object
				// re-acquired a weak reference in its destructor
				if (FFL_atomicValueEqual(&(mRefs->mWeak),0)) {
					delete mRefs;
				}
			}
		}
		// for debugging purposes, clear this.
		const_cast<weakref_impl*&>(mRefs) = NULL;
	}

	void RefBase::extendObjectLifetime(int32_t mode)
	{		
		FFL_atomicSet(&mRefs->mFlags, mode);
	}

	void RefBase::onFirstRef()
	{
	}

	void RefBase::onLastStrongRef(const void* /*id*/)
	{
	}

	bool RefBase::onIncStrongAttempted(uint32_t flags, const void* id)
	{
		return (flags&FIRST_INC_STRONG) ? true : false;
	}

	void RefBase::onLastWeakRef(const void* /*id*/)
	{
	}

	void RefBase::incStrong(const void* id) const
	{
		weakref_impl* const refs = mRefs;
		refs->incWeak(id);

		refs->addStrongRef(id);
		const int32_t c = FFL_atomicInc(&refs->mStrong);
		ALOG_ASSERT(c > 0, "incStrong() called on %p after last strong ref", refs);
#if PRINT_REFS
		ALOGD("incStrong of %p from %p: cnt=%d\n", this, id, c);
#endif
		if (c != INITIAL_STRONG_VALUE) {
			return;
		}

		FFL_atomicAdd(&refs->mStrong, -INITIAL_STRONG_VALUE);
		refs->mBase->onFirstRef();
	}

	void RefBase::decStrong(const void* id) const
	{
		weakref_impl* const refs = mRefs;
		refs->removeStrongRef(id);
		const int32_t c = FFL_atomicDec(&refs->mStrong);
#if PRINT_REFS
		ALOGD("decStrong of %p from %p: cnt=%d\n", this, id, c);
#endif
		ALOG_ASSERT(c >= 1, "decStrong() called on %p too many times", refs);
		if (c == 1) {
			refs->mBase->onLastStrongRef(id);
			if (FFL_atomicValueAnd(&(refs->mFlags) ,OBJECT_LIFETIME_MASK)
				== OBJECT_LIFETIME_STRONG) {
				delete this;
			}
		}
		refs->decWeak(id);
	}

	void RefBase::forceIncStrong(const void* id) const
	{
		weakref_impl* const refs = mRefs;
		refs->incWeak(id);

		refs->addStrongRef(id);
		const int32_t c = FFL_atomicInc(&refs->mStrong);
		ALOG_ASSERT(c >= 0, "forceIncStrong called on %p after ref count underflow",
			refs);
#if PRINT_REFS
		ALOGD("forceIncStrong of %p from %p: cnt=%d\n", this, id, c);
#endif

		switch (c) {
		case INITIAL_STRONG_VALUE:
			FFL_atomicAdd(&refs->mStrong,-INITIAL_STRONG_VALUE);
			// fall through...
		case 0:
			refs->mBase->onFirstRef();
		}
	}

	int32_t RefBase::getStrongCount() const
	{
		return FFL_atomicValueGet(&(mRefs->mStrong));		
	}

	RefBase* weakref_type::refBase() const
	{
		return static_cast<const weakref_impl*>(this)->mBase;
	}

	void weakref_type::incWeak(const void* id)
	{
		weakref_impl* const impl = static_cast<weakref_impl*>(this);
		impl->addWeakRef(id);
		const int32_t c = FFL_atomicInc(&impl->mWeak);
		ALOG_ASSERT(c >= 0, "incWeak called on %p after last weak ref", this);
	}


	void weakref_type::decWeak(const void* id)
	{
		weakref_impl* const impl = static_cast<weakref_impl*>(this);
		impl->removeWeakRef(id);
		const int32_t c = FFL_atomicDec(&impl->mWeak);
		ALOG_ASSERT(c >= 1, "decWeak called on %p too many times", this);
		if (c != 1) return;

		if (FFL_atomicValueAnd(&(impl->mFlags), RefBase::OBJECT_LIFETIME_WEAK)
			== RefBase::OBJECT_LIFETIME_STRONG) {
			// This is the regular lifetime case. The object is destroyed
			// when the last strong reference goes away. Since weakref_impl
			// outlive the object, it is not destroyed in the dtor, and
			// we'll have to do it here.
			if (FFL_atomicValueEqual(&(impl->mStrong),INITIAL_STRONG_VALUE)) {
				// Special case: we never had a strong reference, so we need to
				// destroy the object now.
				delete impl->mBase;
			}
			else {
				// ALOGV("Freeing refs %p of old RefBase %p\n", this, impl->mBase);
				delete impl;
			}
		}
		else {
			// less common case: lifetime is OBJECT_LIFETIME_{WEAK|FOREVER}
			impl->mBase->onLastWeakRef(id);
			if (FFL_atomicValueAnd(&(impl->mFlags),RefBase::OBJECT_LIFETIME_MASK)
				== RefBase::OBJECT_LIFETIME_WEAK) {
				// this is the OBJECT_LIFETIME_WEAK case. The last weak-reference
				// is gone, we can destroy the object.
				delete impl->mBase;
			}
		}
	}

	bool weakref_type::attemptIncStrong(const void* id)
	{
		incWeak(id);

		weakref_impl* impl = static_cast<weakref_impl*>(this);
		int32_t curCount = FFL_atomicValueGet(&(impl->mStrong));

		ALOG_ASSERT(curCount >= 0,
			"attemptIncStrong called on %p after underflow", this);

		while (curCount > 0 && curCount != INITIAL_STRONG_VALUE) {
			// we're in the easy/common case of promoting a weak-reference
			// from an existing strong reference.
			if (FFL_atomicCmpxchg(&impl->mStrong,curCount, curCount + 1) == 0) {
				break;
			}
			// the strong count has changed on us, we need to re-assert our
			// situation.
			curCount =FFL_atomicValueGet(&(impl->mStrong));
		}

		if (curCount <= 0 || curCount == INITIAL_STRONG_VALUE) {
			// we're now in the harder case of either:
			// - there never was a strong reference on us
			// - or, all strong references have been released
			if (FFL_atomicValueAnd(&(impl->mFlags),RefBase::OBJECT_LIFETIME_WEAK)
				== RefBase::OBJECT_LIFETIME_STRONG) {
				// this object has a "normal" life-time, i.e.: it gets destroyed
				// when the last strong reference goes away
				if (curCount <= 0) {
					// the last strong-reference got released, the object cannot
					// be revived.
					decWeak(id);
					return false;
				}

				// here, curCount == INITIAL_STRONG_VALUE, which means
				// there never was a strong-reference, so we can try to
				// promote this object; we need to do that atomically.
				while (curCount > 0) {
					if (FFL_atomicCmpxchg(&impl->mStrong,curCount, curCount + 1) == 0) {
						break;
					}
					// the strong count has changed on us, we need to re-assert our
					// situation (e.g.: another thread has inc/decStrong'ed us)
					curCount = FFL_atomicValueGet(&(impl->mStrong));
				}

				if (curCount <= 0) {
					// promote() failed, some other thread destroyed us in the
					// meantime (i.e.: strong count reached zero).
					decWeak(id);
					return false;
				}
			}
			else {
				// this object has an "extended" life-time, i.e.: it can be
				// revived from a weak-reference only.
				// Ask the object's implementation if it agrees to be revived
				if (!impl->mBase->onIncStrongAttempted(RefBase::FIRST_INC_STRONG, id)) {
					// it didn't so give-up.
					decWeak(id);
					return false;
				}
				// grab a strong-reference, which is always safe due to the
				// extended life-time.
				curCount = FFL_atomicInc(&impl->mStrong);
			}

			// If the strong reference count has already been incremented by
			// someone else, the implementor of onIncStrongAttempted() is holding
			// an unneeded reference.  So call onLastStrongRef() here to remove it.
			// (No, this is not pretty.)  Note that we MUST NOT do this if we
			// are in fact acquiring the first reference.
			if (curCount > 0 && curCount < INITIAL_STRONG_VALUE) {
				impl->mBase->onLastStrongRef(id);
			}
		}

		impl->addStrongRef(id);

#if PRINT_REFS
		ALOGD("attemptIncStrong of %p from %p: cnt=%d\n", this, id, curCount);
#endif

		// now we need to fix-up the count if it was INITIAL_STRONG_VALUE
		// this must be done safely, i.e.: handle the case where several threads
		// were here in attemptIncStrong().
		curCount = FFL_atomicValueGet(&(impl->mStrong));
		while (curCount >= INITIAL_STRONG_VALUE) {
			ALOG_ASSERT(curCount > INITIAL_STRONG_VALUE,
				"attemptIncStrong in %p underflowed to INITIAL_STRONG_VALUE",
				this);
			if (FFL_atomicCmpxchg(&impl->mStrong,curCount, curCount - INITIAL_STRONG_VALUE) == 0) {
				break;
			}
			// the strong-count changed on us, we need to re-assert the situation,
			// for e.g.: it's possible the fix-up happened in another thread.
			curCount = FFL_atomicValueGet(&(impl->mStrong));
		}

		return true;
	}

	bool weakref_type::attemptIncWeak(const void* id)
	{
		weakref_impl* const impl = static_cast<weakref_impl*>(this);

		int32_t curCount = FFL_atomicValueGet(&( impl->mWeak));
		ALOG_ASSERT(curCount >= 0, "attemptIncWeak called on %p after underflow",
			this);
		while (curCount > 0) {
			if (FFL_atomicCmpxchg(&impl->mWeak,curCount, curCount + 1) == 0) {
				break;
			}
			curCount = FFL_atomicValueGet(&(impl->mWeak));
		}

		if (curCount > 0) {
			impl->addWeakRef(id);
		}

		return curCount > 0;
	}

	int32_t weakref_type::getWeakCount() const
	{
		return FFL_atomicValueGet(&( static_cast<const weakref_impl*>(this)->mWeak));
	}

	void weakref_type::printRefs() const
	{
		static_cast<const weakref_impl*>(this)->printRefs();
	}

	void weakref_type::trackMe(bool enable, bool retain)
	{
		static_cast<weakref_impl*>(this)->trackMe(enable, retain);
	}

	weakref_type* RefBase::createWeak(const void* id) const
	{
		mRefs->incWeak(id);
		return mRefs;
	}

	weakref_type* RefBase::getWeakRefs() const
	{
		return mRefs;
	}
	
};

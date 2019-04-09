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
#ifndef _FFL_REF_BASE_H_
#define _FFL_REF_BASE_H_

#include <FFL_Core.h>
#include "FFL_RefWp.hpp"
#include "FFL_RefSp.hpp"

namespace FFL {
	
	class weakref_impl;

class FFLIB_API_IMPORT_EXPORT RefBase
{
	friend class weakref_type;
protected:
	RefBase();
	virtual                 ~RefBase();

	//! Flags for extendObjectLifetime()
	enum 
	{
		//
		//指针的声明周期由强引用计数控制还是弱引用计数控制
		//
		OBJECT_LIFETIME_STRONG = 0x0000,
		OBJECT_LIFETIME_WEAK = 0x0001,
		OBJECT_LIFETIME_MASK = 0x0001
	};

	//
	//指定生命周期强还是弱
	//需要初始化的时候指定
	//
	void            extendObjectLifetime(int32_t mode);

	//! Flags for onIncStrongAttempted()
	enum 
	{
		FIRST_INC_STRONG = 0x0001
	};

	virtual void            onFirstRef();
	virtual void            onLastStrongRef(const void* id);
	virtual bool            onIncStrongAttempted(uint32_t flags, const void* id);
	virtual void            onLastWeakRef(const void* id);
public:
            void            incStrong(const void* id) const;
            void            decStrong(const void* id) const;

            void            forceIncStrong(const void* id) const;

            //! DEBUGGING ONLY: Get current strong ref count.
            int32_t         getStrongCount() const;

   

            weakref_type*   createWeak(const void* id) const;

            weakref_type*   getWeakRefs() const;

            //! DEBUGGING ONLY: Print references held on object.
    inline  void            printRefs() const { getWeakRefs()->printRefs(); }

            //! DEBUGGING ONLY: Enable tracking of object.
    inline  void            trackMe(bool enable, bool retain)
    {
        getWeakRefs()->trackMe(enable, retain);
    }

    typedef RefBase basetype;

private:
    //friend class ReferenceMover;
    //static void moveReferences(void* d, void const* s, size_t n,
    //        const ReferenceConverterBase& caster);
private:
    friend class weakref_type;
    

                            RefBase(const RefBase& o);
            RefBase&        operator=(const RefBase& o);

    weakref_impl* const mRefs;
};
template class FFLIB_API_IMPORT_EXPORT FFL::sp<RefBase>;
}; 

#endif 

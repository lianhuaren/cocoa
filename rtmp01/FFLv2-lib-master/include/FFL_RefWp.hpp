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
*  弱引用
*/


#ifndef _FFL_REF_WP_H_
#define _FFL_REF_WP_H_

#include "FFL_RefSp.hpp"

// ---------------------------------------------------------------------------

namespace FFL {
	// ---------------------------------------------------------------------------
	//比较操作符定义，参数分为强指针，原始指针，其他派生类强指针， 其他派生类原始指针
	//
	#define COMPARE_WEAK(_op_)                                      \
	inline bool operator _op_ (const sp<T>& o) const {              \
		return m_ptr _op_ o.m_ptr;                                  \
	}                                                               \
	inline bool operator _op_ (const T* o) const {                  \
		return m_ptr _op_ o;                                        \
	}                                                               \
	template<typename U>                                            \
	inline bool operator _op_ (const sp<U>& o) const {              \
		return m_ptr _op_ o.m_ptr;                                  \
	}                                                               \
	template<typename U>                                            \
	inline bool operator _op_ (const U* o) const {                  \
		return m_ptr _op_ o;                                        \
	}
    
    class RefBase;
    class FFLIB_API_IMPORT_EXPORT weakref_type
    {
    public:
        RefBase*            refBase() const;
        
        void                incWeak(const void* id);
        void                decWeak(const void* id);
        
        // acquires a strong reference if there is already one.
        bool                attemptIncStrong(const void* id);
        
        // acquires a weak reference if there is already one.
        // This is not always safe. see ProcessState.cpp and BpBinder.cpp
        // for proper use.
        bool                attemptIncWeak(const void* id);
        
        //! DEBUGGING ONLY: Get current weak ref count.
        int32_t             getWeakCount() const;
        
        //! DEBUGGING ONLY: Print references held on object.
        void                printRefs() const;
        
        //! DEBUGGING ONLY: Enable tracking for this object.
        // enable -- enable/disable tracking
        // retain -- when tracking is enable, if true, then we save a stack trace
        //           for each reference and dereference; when retain == false, we
        //           match up references and dereferences and keep only the
        //           outstanding ones.
        
        void                trackMe(bool enable, bool retain);
    };

 
template <typename T>
class FFLIB_API_IMPORT_EXPORT wp
{
public:
    inline wp() : m_ptr(0) { }

    wp(T* other);
    wp(const wp<T>& other);
    wp(const sp<T>& other);
    template<typename U> wp(U* other);
    template<typename U> wp(const sp<U>& other);
    template<typename U> wp(const wp<U>& other);

    ~wp();    

    wp& operator = (T* other);
    wp& operator = (const wp<T>& other);
    wp& operator = (const sp<T>& other);

    template<typename U> wp& operator = (U* other);
    template<typename U> wp& operator = (const wp<U>& other);
    template<typename U> wp& operator = (const sp<U>& other);

    void set_object_and_refs(T* other, weakref_type* refs);

    // promotion to sp

    sp<T> promote() const;

    // Reset

    void clear();

    // Accessors

    inline  weakref_type* get_refs() const { return m_refs; }

    inline  T* unsafe_get() const { return m_ptr; }

    // Operators

    COMPARE_WEAK(==)
    COMPARE_WEAK(!=)
    COMPARE_WEAK(>)
    COMPARE_WEAK(<)
    COMPARE_WEAK(<=)
    COMPARE_WEAK(>=)

    inline bool operator == (const wp<T>& o) const {
        return (m_ptr == o.m_ptr) && (m_refs == o.m_refs);
    }
    template<typename U>
    inline bool operator == (const wp<U>& o) const {
        return m_ptr == o.m_ptr;
    }

    inline bool operator > (const wp<T>& o) const {
        return (m_ptr == o.m_ptr) ? (m_refs > o.m_refs) : (m_ptr > o.m_ptr);
    }
    template<typename U>
    inline bool operator > (const wp<U>& o) const {
        return (m_ptr == o.m_ptr) ? (m_refs > o.m_refs) : (m_ptr > o.m_ptr);
    }

    inline bool operator < (const wp<T>& o) const {
        return (m_ptr == o.m_ptr) ? (m_refs < o.m_refs) : (m_ptr < o.m_ptr);
    }
    template<typename U>
    inline bool operator < (const wp<U>& o) const {
        return (m_ptr == o.m_ptr) ? (m_refs < o.m_refs) : (m_ptr < o.m_ptr);
    }
                         inline bool operator != (const wp<T>& o) const { return m_refs != o.m_refs; }
    template<typename U> inline bool operator != (const wp<U>& o) const { return !operator == (o); }
                         inline bool operator <= (const wp<T>& o) const { return !operator > (o); }
    template<typename U> inline bool operator <= (const wp<U>& o) const { return !operator > (o); }
                         inline bool operator >= (const wp<T>& o) const { return !operator < (o); }
    template<typename U> inline bool operator >= (const wp<U>& o) const { return !operator < (o); }

private:
    template<typename Y> friend class sp;
    template<typename Y> friend class wp;

    T*              m_ptr;
    weakref_type*   m_refs;
};

#undef COMPARE_WEAK


template<typename T>
wp<T>::wp(T* other)
    : m_ptr(other)
{
    if (other) m_refs = other->createWeak(this);
}

template<typename T>
wp<T>::wp(const wp<T>& other)
    : m_ptr(other.m_ptr), m_refs(other.m_refs)
{
    if (m_ptr) m_refs->incWeak(this);
}

template<typename T>
wp<T>::wp(const sp<T>& other)
    : m_ptr(other.m_ptr)
{
    if (m_ptr) {
        m_refs = m_ptr->createWeak(this);
    }
}

template<typename T> template<typename U>
wp<T>::wp(U* other)
    : m_ptr(other)
{
    if (other) m_refs = other->createWeak(this);
}

template<typename T> template<typename U>
wp<T>::wp(const wp<U>& other)
    : m_ptr(other.m_ptr)
{
    if (m_ptr) {
        m_refs = other.m_refs;
        m_refs->incWeak(this);
    }
}

template<typename T> template<typename U>
wp<T>::wp(const sp<U>& other)
    : m_ptr(other.m_ptr)
{
    if (m_ptr) {
        m_refs = m_ptr->createWeak(this);
    }
}

template<typename T>
wp<T>::~wp()
{
    if (m_ptr) m_refs->decWeak(this);
}

template<typename T>
wp<T>& wp<T>::operator = (T* other)
{
    weakref_type* newRefs =
        other ? other->createWeak(this) : 0;
    if (m_ptr) m_refs->decWeak(this);
    m_ptr = other;
    m_refs = newRefs;
    return *this;
}

template<typename T>
wp<T>& wp<T>::operator = (const wp<T>& other)
{
    weakref_type* otherRefs(other.m_refs);
    T* otherPtr(other.m_ptr);
    if (otherPtr) otherRefs->incWeak(this);
    if (m_ptr) m_refs->decWeak(this);
    m_ptr = otherPtr;
    m_refs = otherRefs;
    return *this;
}

template<typename T>
wp<T>& wp<T>::operator = (const sp<T>& other)
{
    weakref_type* newRefs =
        other != NULL ? other->createWeak(this) : 0;
    T* otherPtr(other.m_ptr);
    if (m_ptr) m_refs->decWeak(this);
    m_ptr = otherPtr;
    m_refs = newRefs;
    return *this;
}

template<typename T> template<typename U>
wp<T>& wp<T>::operator = (U* other)
{
    weakref_type* newRefs =
        other ? other->createWeak(this) : 0;
    if (m_ptr) m_refs->decWeak(this);
    m_ptr = other;
    m_refs = newRefs;
    return *this;
}

template<typename T> template<typename U>
wp<T>& wp<T>::operator = (const wp<U>& other)
{
    weakref_type* otherRefs(other.m_refs);
    U* otherPtr(other.m_ptr);
    if (otherPtr) otherRefs->incWeak(this);
    if (m_ptr) m_refs->decWeak(this);
    m_ptr = otherPtr;
    m_refs = otherRefs;
    return *this;
}

template<typename T> template<typename U>
wp<T>& wp<T>::operator = (const sp<U>& other)
{
    weakref_type* newRefs =
        other != NULL ? other->createWeak(this) : 0;
    U* otherPtr(other.m_ptr);
    if (m_ptr) m_refs->decWeak(this);
    m_ptr = otherPtr;
    m_refs = newRefs;
    return *this;
}

template<typename T>
void wp<T>::set_object_and_refs(T* other, weakref_type* refs)
{
    if (other) refs->incWeak(this);
    if (m_ptr) m_refs->decWeak(this);
    m_ptr = other;
    m_refs = refs;
}

template<typename T>
sp<T> wp<T>::promote() const
{
    sp<T> result;
    if (m_ptr && m_refs->attemptIncStrong(&result)) {
        result.set_pointer(m_ptr);
    }
    return result;
}

template<typename T>
void wp<T>::clear()
{
    if (m_ptr) {
        m_refs->decWeak(this);
        m_ptr = 0;
    }
}

}; 

#endif 

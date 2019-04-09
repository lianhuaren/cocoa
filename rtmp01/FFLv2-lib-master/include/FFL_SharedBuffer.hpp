/*
 * Copyright (C) 2005 The Android Open Source Project
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

 /*
 *  This file is part of FFL project.
 *
 *  The MIT License (MIT)
 *  Copyright (C) 2017-2018 zhufeifei All rights reserved.
 *
 *  FFL_SharedBuffer.hpp
 *  Created by zhufeifei(34008081@qq.com) on 2018/12/10
 *  https://github.com/zhenfei2016/FFL-v2.git
 *
 *  移植修改一些内容
 */

#ifndef _FFL_SHARED_BUFFER_HPP_
#define _FFL_SHARED_BUFFER_HPP_

#include <FFL_Core.h>

namespace FFL {

class FFLIB_API_IMPORT_EXPORT SharedBuffer
{
public:

    /* flags to use with release() */
    enum {
        eKeepStorage = 0x00000001
    };

    /*! allocate a buffer of size 'size' and acquire() it.
     *  call release() to free it.
     */
    static          SharedBuffer*           alloc(size_t size);
    
    /*! free the memory associated with the SharedBuffer.
     * Fails if there are any users associated with this SharedBuffer.
     * In other words, the buffer must have been release by all its
     * users.
     */
    static          int                 dealloc(const SharedBuffer* released);
    
    //! get the SharedBuffer from the data pointer
    static  inline  const SharedBuffer*     sharedBuffer(const void* data);

    //! access the data for read
    inline          const void*             data() const;
    
    //! access the data for read/write
    inline          void*                   data();

    //! get size of the buffer
    inline          size_t                  size() const;
 
    //! get back a SharedBuffer object from its data
    static  inline  SharedBuffer*           bufferFromData(void* data);
    
    //! get back a SharedBuffer object from its data
    static  inline  const SharedBuffer*     bufferFromData(const void* data);

    //! get the size of a SharedBuffer object from its data
    static  inline  size_t                  sizeFromData(const void* data);
    
    //! edit the buffer (get a writtable, or non-const, version of it)
                    SharedBuffer*           edit() const;

    //! edit the buffer, resizing if needed
                    SharedBuffer*           editResize(size_t size) const;

    //! like edit() but fails if a copy is required
                    SharedBuffer*           attemptEdit() const;
    
    //! resize and edit the buffer, loose it's content.
                    SharedBuffer*           reset(size_t size) const;

    //! acquire/release a reference on this buffer
                    void                    acquire() const;
                    
    /*! release a reference on this buffer, with the option of not
     * freeing the memory associated with it if it was the last reference
     * returns the previous reference count
     */     
                    int32_t                 release(uint32_t flags = 0) const;
    
    //! returns wether or not we're the only owner
    inline          bool                    onlyOwner() const;
    

private:
        inline SharedBuffer() { }
        inline ~SharedBuffer() { }
        inline SharedBuffer(const SharedBuffer&);
 
		
        // 16 bytes. must be sized to preserve correct alingment.        
                size_t         mSize;
				uint32_t       mReserved1;
		mutable volatile int32_t        mRefs;
                uint32_t       mReserved2;

		
		//union {
		//	typedef struct{
		//		mutable volatile int32_t mRefCount;
		//		uint32_t  mReserved1;			
		//		uint32_t  mSize;
		//	}mInfo;
		//	uint8_t mBytes[12];
		//} mHeader;		
};

// ---------------------------------------------------------------------------

const SharedBuffer* SharedBuffer::sharedBuffer(const void* data) {
    return data ? reinterpret_cast<const SharedBuffer *>(data)-1 : 0;
}

const void* SharedBuffer::data() const {
    return this + 1;
}

void* SharedBuffer::data() {
    return this + 1;
}

size_t SharedBuffer::size() const {
    return mSize;
}

SharedBuffer* SharedBuffer::bufferFromData(void* data)
{
    return ((SharedBuffer*)data)-1;
}
    
const SharedBuffer* SharedBuffer::bufferFromData(const void* data)
{
    return ((const SharedBuffer*)data)-1;
}

size_t SharedBuffer::sizeFromData(const void* data)
{
    return (((const SharedBuffer*)data)-1)->mSize;
}

bool SharedBuffer::onlyOwner() const {
    return (mRefs == 1);
}

}; 

// ---------------------------------------------------------------------------

#endif // ANDROID_VECTOR_H

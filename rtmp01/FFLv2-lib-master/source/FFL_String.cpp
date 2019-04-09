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
*  FFL_String.cpp
*  Created by zhufeifei(34008081@qq.com) on 2018/12/10
*  https://github.com/zhenfei2016/FFL-v2.git
*
*  为了方便shared模块导出，移植了这个文件
*/

#include <FFL_String.hpp>
#include <FFL_SharedBuffer.hpp>
#include <stdio.h>
#include <ctype.h>
#include <stdlib.h>
#include "internalLogConfig.h"

#define OS_PATH_SEPARATOR  '/'
#define NO_MEMORY -1
#define NO_ERROR 0

#define ALOG_ASSERT 
/*
* Functions outside android is below the namespace android, since they use
* functions and constants in android namespace.
*/

// ---------------------------------------------------------------------------

namespace FFL {

	// Separator used by resource paths. This is not platform dependent contrary
	// to OS_PATH_SEPARATOR.
#define RES_PATH_SEPARATOR '/'

	static SharedBuffer* gEmptyStringBuf = NULL;
	static char* gEmptyString = NULL;

	extern int gDarwinCantLoadAllObjects;
	int gDarwinIsReallyAnnoying;



    extern "C" void initialize_string8(){
		// HACK: This dummy dependency forces linking libutils Static.cpp,
		// which is needed to initialize String8/String16 classes.
		// These variables are named for Darwin, but are needed elsewhere too,
		// including static linking on any platform.
		//gDarwinIsReallyAnnoying = gDarwinCantLoadAllObjects;
		if (!gEmptyStringBuf) {
			SharedBuffer* buf = SharedBuffer::alloc(1);
			char* str = (char*)buf->data();
			*str = 0;
			gEmptyStringBuf = buf;
			gEmptyString = str;
		}
	}

	extern "C" void terminate_string8(){
		if (gEmptyStringBuf) {
			SharedBuffer::bufferFromData(gEmptyString)->release();
			gEmptyStringBuf = NULL;
			gEmptyString = NULL;
		}
	}

	static inline char* getEmptyString()
	{
		if (gEmptyStringBuf == NULL) {
			INTERNAL_FFL_LOG_WARNING("String8 not call initialize_string8()");
			initialize_string8();
		}

		gEmptyStringBuf->acquire();
		return gEmptyString;
	}

	// ---------------------------------------------------------------------------

	static char* allocFromUTF8(const char* in, size_t len)
	{
		if (len > 0) {
			SharedBuffer* buf = SharedBuffer::alloc(len + 1);
			FFL_ASSERT_LOG(buf, "Unable to allocate shared buffer");
			if (buf) {
				char* str = (char*)buf->data();
				memcpy(str, in, len);
				str[len] = 0;
				return str;
			}
			return NULL;
		}

		return getEmptyString();
	}


	// ---------------------------------------------------------------------------

	String8::String8()
		: mString(getEmptyString())
	{
	}

	String8::String8(const String8& o)
		: mString(o.mString)
	{
		SharedBuffer::bufferFromData(mString)->acquire();
	}

	String8::String8(const char* o)
		: mString(allocFromUTF8(o, strlen(o)))
	{
		if (mString == NULL) {
			mString = getEmptyString();
		}
	}

	String8::String8(const char* o, size_t len)
		: mString(allocFromUTF8(o, len))
	{
		if (mString == NULL) {
			mString = getEmptyString();
		}
	}



	String8::~String8()
	{
		SharedBuffer::bufferFromData(mString)->release();
	}

	String8 String8::format(const char* fmt, ...)
	{
		va_list args;
		va_start(args, fmt);

		String8 result(formatV(fmt, args));

		va_end(args);
		return result;
	}

	String8 String8::formatV(const char* fmt, va_list args)
	{
		String8 result;
		result.appendFormatV(fmt, args);
		return result;
	}

	void String8::clear() {
		SharedBuffer::bufferFromData(mString)->release();
		mString = getEmptyString();
	}

	void String8::setTo(const String8& other)
	{
		SharedBuffer::bufferFromData(other.mString)->acquire();
		SharedBuffer::bufferFromData(mString)->release();
		mString = other.mString;
	}

	status_t String8::setTo(const char* other){
		if (other) {
			const char *newString = allocFromUTF8(other, strlen(other));
			SharedBuffer::bufferFromData(mString)->release();
			mString = newString;
			if (mString) return NO_ERROR;
		}

		mString = getEmptyString();
		return NO_MEMORY;
	}

	status_t String8::setTo(const char* other, size_t len)
	{
		const char *newString = allocFromUTF8(other, len);
		SharedBuffer::bufferFromData(mString)->release();
		mString = newString;
		if (mString) return NO_ERROR;

		mString = getEmptyString();
		return NO_MEMORY;
	}



	status_t String8::append(const String8& other)
	{
		const size_t otherLen = other.bytes();
		if (bytes() == 0) {
			setTo(other);
			return NO_ERROR;
		}
		else if (otherLen == 0) {
			return NO_ERROR;
		}

		return real_append(other.string(), otherLen);
	}

	status_t String8::append(const char* other)
	{
		return append(other, strlen(other));
	}

	status_t String8::append(const char* other, size_t otherLen)
	{
		if (bytes() == 0) {
			return setTo(other, otherLen);
		}
		else if (otherLen == 0) {
			return NO_ERROR;
		}

		return real_append(other, otherLen);
	}

	status_t String8::appendFormat(const char* fmt, ...)
	{
		va_list args;
		va_start(args, fmt);

		status_t result = appendFormatV(fmt, args);

		va_end(args);
		return result;
	}

	status_t String8::appendFormatV(const char* fmt, va_list args)
	{
		int result = NO_ERROR;
		int n = vsnprintf(NULL, 0, fmt, args);
		if (n != 0) {
			size_t oldLength = length();
			char* buf = lockBuffer(oldLength + n);
			if (buf) {
				vsnprintf(buf + oldLength, n + 1, fmt, args);
			}
			else {
				result = NO_MEMORY;
			}
		}
		return result;
	}

	status_t String8::real_append(const char* other, size_t otherLen)
	{
		const size_t myLen = bytes();

		SharedBuffer* buf = SharedBuffer::bufferFromData(mString)
			->editResize(myLen + otherLen + 1);
		if (buf) {
			char* str = (char*)buf->data();
			mString = str;
			str += myLen;
			memcpy(str, other, otherLen);
			str[otherLen] = '\0';
			return NO_ERROR;
		}
		return NO_MEMORY;
	}

	char* String8::lockBuffer(size_t size)
	{
		SharedBuffer* buf = SharedBuffer::bufferFromData(mString)
			->editResize(size + 1);
		if (buf) {
			char* str = (char*)buf->data();
			mString = str;
			return str;
		}
		return NULL;
	}

	void String8::unlockBuffer()
	{
		unlockBuffer(strlen(mString));
	}

	status_t String8::unlockBuffer(size_t size)
	{
		if (size != this->size()) {
			SharedBuffer* buf = SharedBuffer::bufferFromData(mString)
				->editResize(size + 1);
			if (!buf) {
				return NO_MEMORY;
			}

			char* str = (char*)buf->data();
			str[size] = 0;
			mString = str;
		}

		return NO_ERROR;
	}

	int String8::find(const char* other, size_t start) const
	{
		size_t len = size();
		if (start >= len) {
			return -1;
		}
		const char* s = mString + start;
		const char* p = strstr(s, other);
		return p ? p - mString : -1;
	}

	void String8::toLower()
	{
		toLower(0, size());
	}

	void String8::toLower(size_t start, size_t length)
	{
		const size_t len = size();
		if (start >= len) {
			return;
		}
		if (start + length > len) {
			length = len - start;
		}
		char* buf = lockBuffer(len);
		buf += start;
		while (length > 0) {
			*buf = tolower(*buf);
			buf++;
			length--;
		}
		unlockBuffer(len);
	}

	void String8::toUpper()
	{
		toUpper(0, size());
	}

	void String8::toUpper(size_t start, size_t length)
	{
		const size_t len = size();
		if (start >= len) {
			return;
		}
		if (start + length > len) {
			length = len - start;
		}
		char* buf = lockBuffer(len);
		buf += start;
		while (length > 0) {
			*buf = toupper(*buf);
			buf++;
			length--;
		}
		unlockBuffer(len);
	}


	// ---------------------------------------------------------------------------
	// Path functions

	void String8::setPathName(const char* name)
	{
		setPathName(name, strlen(name));
	}

	void String8::setPathName(const char* name, size_t len)
	{
		char* buf = lockBuffer(len);

		memcpy(buf, name, len);

		// remove trailing path separator, if present
		if (len > 0 && buf[len - 1] == OS_PATH_SEPARATOR)
			len--;

		buf[len] = '\0';

		unlockBuffer(len);
	}

	String8 String8::getPathLeaf(void) const
	{
		const char* cp;
		const char*const buf = mString;

		cp = strrchr(buf, OS_PATH_SEPARATOR);
		if (cp == NULL)
			return String8(*this);
		else
			return String8(cp + 1);
	}

	String8 String8::getPathDir(void) const
	{
		const char* cp;
		const char*const str = mString;

		cp = strrchr(str, OS_PATH_SEPARATOR);
		if (cp == NULL)
			return String8("");
		else
			return String8(str, cp - str);
	}

	String8 String8::walkPath(String8* outRemains) const
	{
		const char* cp;
		const char*const str = mString;
		const char* buf = str;

		cp = strchr(buf, OS_PATH_SEPARATOR);
		if (cp == buf) {
			// don't include a leading '/'.
			buf = buf + 1;
			cp = strchr(buf, OS_PATH_SEPARATOR);
		}

		if (cp == NULL) {
			String8 res = buf != str ? String8(buf) : *this;
			if (outRemains) *outRemains = String8("");
			return res;
		}

		String8 res(buf, cp - buf);
		if (outRemains) *outRemains = String8(cp + 1);
		return res;
	}

	/*
	* Helper function for finding the start of an extension in a pathname.
	*
	* Returns a pointer inside mString, or NULL if no extension was found.
	*/
	char* String8::find_extension(void) const
	{
		const char* lastSlash;
		const char* lastDot;
		const char* const str = mString;

		// only look at the filename
		lastSlash = strrchr(str, OS_PATH_SEPARATOR);
		if (lastSlash == NULL)
			lastSlash = str;
		else
			lastSlash++;

		// find the last dot
		lastDot = strrchr(lastSlash, '.');
		if (lastDot == NULL)
			return NULL;

		// looks good, ship it
		return const_cast<char*>(lastDot);
	}

	String8 String8::getPathExtension(void) const
	{
		char* ext;

		ext = find_extension();
		if (ext != NULL)
			return String8(ext);
		else
			return String8("");
	}

	String8 String8::getBasePath(void) const
	{
		char* ext;
		const char* const str = mString;

		ext = find_extension();
		if (ext == NULL)
			return String8(*this);
		else
			return String8(str, ext - str);
	}

	String8& String8::appendPath(const char* name)
	{
		// TODO: The test below will fail for Win32 paths. Fix later or ignore.
		if (name[0] != OS_PATH_SEPARATOR) {
			if (*name == '\0') {
				// nothing to do
				return *this;
			}

			size_t len = length();
			if (len == 0) {
				// no existing filename, just use the new one
				setPathName(name);
				return *this;
			}

			// make room for oldPath + '/' + newPath
			int newlen = strlen(name);

			char* buf = lockBuffer(len + 1 + newlen);

			// insert a '/' if needed
			if (buf[len - 1] != OS_PATH_SEPARATOR)
				buf[len++] = OS_PATH_SEPARATOR;

			memcpy(buf + len, name, newlen + 1);
			len += newlen;

			unlockBuffer(len);

			return *this;
		}
		else {
			setPathName(name);
			return *this;
		}
	}

	String8& String8::convertToResPath()
	{
#if OS_PATH_SEPARATOR != RES_PATH_SEPARATOR
		size_t len = length();
		if (len > 0) {
			char * buf = lockBuffer(len);
			for (char * end = buf + len; buf < end; ++buf) {
				if (*buf == OS_PATH_SEPARATOR)
					*buf = RES_PATH_SEPARATOR;
			}
			unlockBuffer(len);
		}
#endif
		return *this;
	}
	
	int64_t  String8::toInt64() {
		if (mString) {
#if WIN32
			return ::_atoi64(mString);
#else
			return (int64_t)::atoll(mString);
#endif
		}
		return 0;
	}
	int32_t  String8::toInt32() {
		if (mString) {
			return ::atoi(mString);
		}
		return 0;
	}
	//
	//  ʲô��ʼ������
	bool String8::startWith(const char* sub) {
		if (sub == NULL) {
			return false;
		}

		size_t subLength= strlen(sub);
		if (length() == 0 && subLength == 0) {
			return true;
		}

		if (subLength > length()) {
			return true;
		}

		return memcmp(string(), sub, subLength)==0;
	}
	bool String8::endWith(const char*sub){
		if (sub == NULL) {
			return false;
		}

		size_t subLength = strlen(sub);
		if (length() == 0 && subLength == 0) {
			return true;
		}

		if (subLength > length()) {
			return true;
		}

		return memcmp(string()+size() - subLength, sub, subLength) == 0;
	}
	bool String8::equal(const char* other) {
		if (other == NULL || other[0] == 0) {
			return size() == 0;
		}

		size_t otherLength = strlen(other);
		if (otherLength != size()) {
			return false;
		}

		return memcmp(string(), other, otherLength) == 0;
	}
	bool String8::equal(const String8& other) {
		return equal(other.string());
	}
	bool String8::equalIgnoreCase(const char* other) {
		if (other == NULL || other[0] == 0) {
			return size() == 0;
		}

		size_t otherLength = strlen(other);
		if (otherLength != size()) {
			return false;
		}

		uint8_t* p1 =(uint8_t*)other;
		uint8_t* p2 = (uint8_t*)string();
		for(size_t i=0;i<otherLength;i++){
			if (p1[i] == p2[i]) {
				continue;
			}else if (p1[i] > p2[i]) {
				if (p1[i] - p2[i] == 0x20) {
					continue;
				}
			}else  {
				if (p2[i] - p1[i] == 0x20) {
					continue;
				}
			}
			return false;
		}
		
		return true;
	}
	bool String8::equalIgnoreCase(const String8& other) {
		return equalIgnoreCase(other.string());
	}

	size_t String8::length() const
	{
		return SharedBuffer::sizeFromData(mString) - 1;
	}

	size_t String8::bytes() const
	{
		return SharedBuffer::sizeFromData(mString) - 1;
	}

	const SharedBuffer* String8::sharedBuffer() const
	{
		return SharedBuffer::bufferFromData(mString);
	}
};

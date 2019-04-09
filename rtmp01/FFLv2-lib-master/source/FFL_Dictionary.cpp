/*
*  This file is part of FFL project.
*
*  The MIT License (MIT)
*  Copyright (C) 2017-2018 zhufeifei All rights reserved.
*
*  FFL_Dictionary.cpp   
*  Created by zhufeifei(34008081@qq.com) on 2018/06/24 
*  https://github.com/zhenfei2016/FFLv2-lib.git
*
*
*/
#include <FFL_String.hpp>
#include <FFL_Dictionary.hpp>
#include <map>
#include <FFL_Core.h>

namespace FFL{
	typedef std::map<String, String> DIC_MAP;
	typedef DIC_MAP::iterator PDIC_MAP;

	class DictionaryImpl {
	public:
		DictionaryImpl(){}
		~DictionaryImpl(){}

		DIC_MAP mDic;
	};

	Dictionary::Dictionary(){
		mImpl = new DictionaryImpl();
	}
	Dictionary::~Dictionary(){
		FFL_SafeFree(mImpl);
	}
	Dictionary::Dictionary(const Dictionary & r) {
		*mImpl = *r.mImpl;
	}
	Dictionary & Dictionary::operator=(const Dictionary & r) {
		mImpl = new DictionaryImpl();
		*mImpl = *(r.mImpl);
		return *this;
	}
	//
	//  写一条key/value
	//
	void Dictionary::setKey(const String& key, const String& value){	
		mImpl->mDic[key] = value;
	}
	void Dictionary::setKey(const char* key, const char* value) {
		String k(key);
		mImpl->mDic[k] = value;
	}
	//
	// 通过key获取值,返回是否找到了这个key
	//
	bool Dictionary::getKey(const String& key, String& value) {
		PDIC_MAP pos = mImpl->mDic.find(key);
		if (pos != mImpl->mDic.end()) {
			value= pos->second;
			return true;
		}
		return false;
	}
	bool Dictionary::getKey(const char* key, String& value) {
		String k(key);
		PDIC_MAP pos = mImpl->mDic.find(k);
		if (pos != mImpl->mDic.end()) {
			value = pos->second;
			return true;
		}
		return false;
	}

	//
	// 获取多小组数据
	int32_t Dictionary::getCount() {
		return (int32_t)mImpl->mDic.size();
	}
	//
	//获取数据到数组中
	bool Dictionary::getAll(Pair* pairArray, int32_t* bufSize) {
		if (pairArray == NULL || (bufSize && *bufSize < getCount())) {
			if (bufSize) {
				*bufSize = getCount();
			}
			return false;
		}

		int i = 0;
		PDIC_MAP pos = mImpl->mDic.begin();
		for (; pos != mImpl->mDic.end(); pos++) {
			pairArray[i].key=pos->first;
			pairArray[i].value = pos->second;
			i++;
		}

		if (bufSize) {
			*bufSize = i;
		}
		return true;
	}
	//
	// 移除一条
	//
	bool Dictionary::removeKey(const char* key) {

		String k(key);
		PDIC_MAP pos = mImpl->mDic.find(k);
		if (pos != mImpl->mDic.end()) {
			mImpl->mDic.erase(pos);
			return true;
		}
		return false;
	}

	void Dictionary::setKeyInt64(const String& key, int64_t value) {
		setKey(key, String::format("%" lld64, value));
	}
	void Dictionary::setKeyInt64(const char* key, int64_t value) {
		setKey(key, String::format("%" lld64, value));
	}
	bool Dictionary::getKeyInt64(const String& key, int64_t& value, int64_t def) {
		String strVal;
		if (getKey(key, strVal)) {
			value = strVal.toInt64();
			return true;
		}
		value = def;
		return false;
	}
	bool Dictionary::getKeyInt64(const char* key, int64_t& value, int64_t def) {
		String strVal;
		if (getKey(key, strVal)) {
			value = strVal.toInt64();
			return true;
		}
		value = def;
		return false;
	}

	void Dictionary::setKeyInt32(const String& key, int32_t value) {		
		setKey(key, String::format("%d", value));
	}
	void Dictionary::setKeyInt32(const char* key, int32_t value) {
		setKey(key, String::format("%d", value));
	}
	bool Dictionary::getKeyInt32(const String& key, int32_t& value, int32_t def) {
		String strVal;
		if (getKey(key, strVal)) {
			value = strVal.toInt32();
			return true;
		}
		value = def;
		return false;
	}		
	bool Dictionary::getKeyInt32(const char* key, int32_t& value, int32_t def) {
		String strVal;
		if (getKey(key, strVal)) {
			value = strVal.toInt32();
			return true;
		}
		value = def;
		return false;
	}

}


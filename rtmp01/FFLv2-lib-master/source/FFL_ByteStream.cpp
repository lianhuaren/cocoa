/*
*  This file is part of FFL project.
*
*  The MIT License (MIT)
*  Copyright (C) 2017-2018 zhufeifei All rights reserved.
*
*  FFL_ByteStream   
*  Created by zhufeifei(34008081@qq.com) on 2018/05/1
*  https://github.com/zhenfei2016/FFLv2-lib.git
*
*
*/
#include <FFL_ByteStream.hpp>

namespace FFL {
	ByteStreamBase::ByteStreamBase():mMemEndian(0){
		setData(0,0,0);
	}
	ByteStreamBase::~ByteStreamBase() {
	}
	//
	//  设置数据，bytestream将在这个数据集上进行操作
	//
	void ByteStreamBase::setData(uint8_t* data, uint32_t size, uint32_t capticity) {
		setData(data,
			size,
			capticity,
			FFL_isLittleEndian() ? FFL_LITTLE_ENDIAN : FFL_BIG_ENDIAN);
	}

	void ByteStreamBase::setData(uint8_t* data, uint32_t size, uint32_t capticity, int memEndian) {
		FFL_ASSERT(capticity >= size);
		mData = data;
		mDataCapacity = capticity;
		mDataSize = size;
		mMemEndian = memEndian;
	}	
	//
	//  是否系统的字节序与mData中的一致
	//
	bool ByteStreamBase::isSameEndian() {
		if (FFL_isLittleEndian()) {
			return mMemEndian == FFL_LITTLE_ENDIAN;
		}
		return mMemEndian == FFL_BIG_ENDIAN;
	}

	//
	//  order:顺序的，还是反序的
	//
	static inline void  internalCopyBytes(uint8_t* s, uint8_t* d, uint32_t size, bool order) {
		if (order) {
			memcpy(d, s, size);
		}
		else {
			for (uint32_t i = 0; i < size; i++) {
				d[size - i - 1] = s[i];
			}
		}
	}	

	ByteStream::ByteStream():mReadPos(0),mWritePos(0){

	}
	ByteStream::~ByteStream() {
	}
	//
	//  重置读写指针
	//
	void ByteStream::reset(){
		mReadPos = 0;
		mWritePos = 0;
		mDataSize = 0;
	}

#define ReadNBytes(n,ret)   \
	if (suc) *suc = false;\
	if (haveData(n)) { \
		readBuffer((uint8_t*)&ret, n, isSameEndian()); \
		if (suc) *suc = true; } \


	//
	//  读写
	//
	int8_t ByteStream::read1Bytes(bool* suc) {
		int8_t val = 0;
		ReadNBytes(1, val);
		return val;
	}

	int16_t ByteStream::read2Bytes(bool* suc) {
		int16_t val = 0;
		ReadNBytes(2, val);
		return val;

	}
	int32_t ByteStream::read3Bytes(bool* suc) {
		int32_t val = 0;
		ReadNBytes(3, val);
		return val;
	}
	int32_t ByteStream::read4Bytes(bool* suc) {
		int32_t val = 0;
		ReadNBytes(4, val);
		return val;
	}
	int64_t ByteStream::read8Bytes(bool* suc) {
		int64_t val = 0;
		ReadNBytes(8, val);
		return val;
	}
	bool ByteStream::readString(String& val, uint32_t len){
		uint32_t size = len;
		if (haveData(size)) {
			uint8_t* buf = new uint8_t[size];
			readBuffer(buf, size, true);
			val.append((const char*)buf, size);
			delete[] buf;
			return true;
		}
		return false;
	}
	bool ByteStream::readBytes(int8_t* val, uint32_t size) {
		if (haveData(size)) {
			readBuffer((uint8_t*)val, size, true);
			return true;
		}
		return false;
	}
	//
	//  跳过多少个字节
	//
	void ByteStream::skipRead(int32_t step) {
		if(step<0 && (uint32_t)(-step) >= mReadPos){
			mReadPos=0;
		}else {
			mReadPos += step;
			if (mReadPos >= mDataSize) {
				mReadPos = mDataSize;
			}
		}
	}
	//
	//  是否还有这么多可以读的数据
	//
	bool ByteStream::haveData(uint32_t size) {
		return (getSize() >= size);
	}

	bool ByteStream::write1Bytes(int8_t val) {
		uint32_t size = 1;
		if (haveSpace(size)) {
			writeBuffer((uint8_t*)&val, size, isSameEndian());
			return true;
		}
		return false;
	}
	bool ByteStream::write2Bytes(int16_t val) {
		uint32_t size = 2;
		if (haveSpace(size)) {
			writeBuffer((uint8_t*)&val, size, isSameEndian());
			return true;
		}
		return false;
	}
	bool ByteStream::write3Bytes(int32_t val) {
		uint32_t size = 3;
		if (haveSpace(size)) {
			writeBuffer((uint8_t*)&val, size, isSameEndian());
			return true;
		}
		return false;
	}
	bool ByteStream::write4Bytes(int32_t val) {
		uint32_t size = 4;
		if (haveSpace(size)) {
			writeBuffer((uint8_t*)&val, size, isSameEndian());
			return true;
		}
		return false;
	}
	bool ByteStream::write8Bytes(int64_t val) {
		uint32_t size = 8;
		if (haveSpace(size)) {
			writeBuffer((uint8_t*)&val, size, isSameEndian());
			return true;
		}
		return false;
	}
	bool ByteStream::writeString(const String& val, uint32_t len) {
		uint32_t size = len;
		if (haveSpace(size)) {
			writeBuffer((uint8_t*)val.string(), size, true);
			return true;
		}
		return false;
	}
	bool ByteStream::writeString(const char* val, uint32_t len) {
		uint32_t size = len;
		if (val && haveSpace(size)) {
			writeBuffer((uint8_t*)val, size, true);
			return true;
		}
		return false;
	}
	bool ByteStream::writeBytes(const int8_t* val, uint32_t size) {
		if (haveSpace(size)) {
			writeBuffer((uint8_t*)val, size, true);
			return true;
		}
		return false;
	}
	//
	//  跳过多少个空位置
	//
	void ByteStream::skipWrite(int32_t step) {
		mWritePos += step;
		if (step < 0) {
			mWritePos = (mWritePos > (uint32_t)(-step))?(mWritePos+step) : 0;
			mDataSize = mWritePos;
			return;
		}

		if (haveSpace(step)) {
			mWritePos += step;
			mDataSize += step;
		}
		else {
			mWritePos = mDataCapacity;
			mDataSize = mDataCapacity;
		}
	}
	//
	//  是否还有这么多空间可以使用
	//
	bool ByteStream::haveSpace(uint32_t size) {
		return getCapacity() >= getSize() + size;
	}	


	void ByteStream::readBuffer(uint8_t* dst, uint32_t size, bool order) {
		FFL_ASSERT(mDataSize >= size);		
		if (dst) {
			internalCopyBytes(getData() + mReadPos, dst, size, order);
		}
		mReadPos += size;
		mDataSize -= size;
	}
	void ByteStream::writeBuffer(uint8_t* src, uint32_t size, bool order) {
		FFL_ASSERT(mWritePos + size <= getCapacity());		
		internalCopyBytes(src,getData() + mWritePos, size, order);
		mWritePos += size;			
		mDataSize += size;
	}
}

/*
*  This file is part of FFL project.
*
*  The MIT License (MIT)
*  Copyright (C) 2017-2018 zhufeifei All rights reserved.
*
*  FFL_NetStream.cpp
*  Created by zhufeifei(34008081@qq.com) on 2018/07/17
*  https://github.com/zhenfei2016/FFLv2-lib.git
*  网络流，读取接口，内部缓存数据，方便进行parser
*
*/
#include <net/FFL_Net.h>
#include <net/FFL_NetStream.hpp>
#include <net/FFL_TcpClient.hpp>
#include <FFL_ByteBuffer.hpp>
#include "internalLogConfig.h"

namespace FFL {
	const static int KBufferSIZE = 4096 * 2;
	NetStreamReader::NetStreamReader(CSocket* socket) :mSocket(socket) {
		mSize = 0;
		mPosition = 0;
		mBuffer = new ByteBuffer(KBufferSIZE );
	}
	NetStreamReader::NetStreamReader(TcpClient* client) {
		mSocket = &(client->mSocket);
		mSize = 0;
		mPosition = 0;
		mBuffer = new ByteBuffer(KBufferSIZE);
	}
	NetStreamReader::~NetStreamReader() {
		FFL_SafeFree(mBuffer);
	}
	uint8_t* NetStreamReader::getData() const {
		return mBuffer->data();
	}
	uint32_t NetStreamReader::getSize() const {
		return mSize;
	}
	uint32_t NetStreamReader::getPosition() const {
		return mPosition;
	}
	void NetStreamReader::skip(int32_t length) {
		mPosition += length;
		mSize -= length;
		FFL_ASSERT(mPosition >= 0 && mPosition < mBuffer->size());
	}
	//
	//  socket拉取数据填充,  expect:期望的大小
	//
	status_t NetStreamReader::pull(int32_t expect) {
		//
		// 如果开始指针到达一半位置了，则移动并拷贝数据到开始
		//
		int32_t spaceSize = mBuffer->size()- mSize ;
		if (expect <= 0 || expect > spaceSize) {
			if (mPosition > 0) {
				uint8_t* src = mBuffer->data() + mPosition;
				uint8_t* dst = mBuffer->data();
				memcpy(dst, src, mSize);
				mPosition = 0;
			}
		}

		if (spaceSize < 0) {
			return FFL_FAILED;
		}

		status_t ret = FFL_OK;		
		size_t readed = 0;
		if (expect <= 0) {
			expect = spaceSize;
		} else {
			expect = FFL_MIN(spaceSize, expect);
		}
		uint8_t* dst = mBuffer->data() + mPosition;
		if (FFL_OK !=(ret= mSocket->read(dst, expect, &readed))) {
			return ret;
		}

		FFL_ASSERT_LOG(((int32_t)readed <= expect), "test buffer size");
		INTERNAL_FFL_LOG_INFO("read data fd=%d  size=%d", mSocket->getFd(),readed);
		mSize += readed;
		return FFL_OK;
	}

#define ReadNBytes(n,ret)   \
	if (suc) *suc = false;\
	if (haveData(n)) { \
		readBuffer((uint8_t*)&ret, n, 0); \
		if (suc) *suc = true; } \

	//
	//  ByteReader 读
	//
	int8_t NetStreamReader::read1Bytes(bool* suc) {
		int8_t ret=0;
		ReadNBytes(1,ret);
		return ret;
	}
	int16_t NetStreamReader::read2Bytes(bool* suc) {
		int16_t ret = 0;
		ReadNBytes(2, ret);
		return ret;
	}
	int32_t NetStreamReader::read3Bytes(bool* suc) {
		int32_t ret = 0;
		ReadNBytes(3, ret);
		return ret;
	}
	int32_t NetStreamReader::read4Bytes(bool* suc) {
		int32_t ret = 0;
		ReadNBytes(4, ret);
		return ret;
	}
	int64_t NetStreamReader::read8Bytes(bool* suc){
		int64_t ret = 0;
		ReadNBytes(8, ret);
		return ret;
	}

	bool NetStreamReader::readString(String& val, uint32_t len) {
		FFL_ASSERT_LOG(0, "not implement");
		return false;
	}
	bool NetStreamReader::readBytes(int8_t* data, uint32_t size) {
		if (size <= mSize) {
			memcpy(data, getData() + getPosition(), size);
			mSize -= size;
			mPosition += size;
			return true;
		}
		return false;
	}
	
	//
	//  跳过多少个字节
	//
	void NetStreamReader::skipRead(int32_t step) {
		if (step <(int32_t) mSize && step>0) {
			mSize -= step;
			mPosition += step;
		}
	}
	//
	//  是否还有这么多可以读的数据
	//
	bool NetStreamReader::haveData(uint32_t size) {
		return mSize>=size;
	}

	void NetStreamReader::readBuffer(uint8_t* dst, uint32_t size, bool order) {
		FFL_ASSERT(mSize >= size);
		if (dst) {
			FFL_copyBytes(getData() + mPosition, dst, size, order);
		}
		mPosition += size;
	}
}
/*
*  This file is part of FFL project.
*
*  The MIT License (MIT)
*  Copyright (C) 2017-2018 zhufeifei All rights reserved.
*
*  FFL_ByteReader   
*  Created by zhufeifei(34008081@qq.com) on 2018/05/01 
*  https://github.com/zhenfei2016/FFLv2-lib.git
*  字节流读接口
*/
#ifndef _FFL_BYTE_READER_HPP_
#define _FFL_BYTE_READER_HPP_

#include <FFL_Core.h>
#include <FFL_String.hpp>

namespace FFL{ 
	class ByteReader {	
	public:
		//
		//  字节流读接口
		//
		virtual int8_t read1Bytes(bool* suc )=0;
		virtual int16_t read2Bytes(bool* suc) = 0;
		virtual int32_t read3Bytes(bool* suc) = 0;
		virtual int32_t read4Bytes(bool* suc) = 0;
		virtual int64_t read8Bytes(bool* suc) = 0;
		virtual bool readString(String& val, uint32_t len) = 0;
		virtual bool readBytes(int8_t* data, uint32_t size) = 0;
		//
		//  跳过多少个字节
		//
		virtual void skipRead(int32_t step) = 0;
		//
		//  是否还有这么多可以读的数据
		//
		virtual bool haveData(uint32_t size) = 0;
	};
}


#endif


/*
*  This file is part of FFL project.
*
*  The MIT License (MIT)
*  Copyright (C) 2017-2018 zhufeifei All rights reserved.
*
*  FFL_NetStream.hpp   
*  Created by zhufeifei(34008081@qq.com) on 2018/07/17 
*  https://github.com/zhenfei2016/FFL-v2.git
*  网络流，读取接口，内部缓存数据，方便进行parser
*
*/
#ifndef _FFL_NET_STREAM_READER_HPP_
#define _FFL_NET_STREAM_READER_HPP_

#include <FFL_ByteReader.hpp>
#include <net/FFL_NetSocket.hpp>

namespace FFL {
	class ByteBuffer;
	class TcpClient;

	class FFLIB_API_IMPORT_EXPORT NetStreamReader : public ByteReader {
	public:
		NetStreamReader(CSocket* socket );
		NetStreamReader(TcpClient* client);
        virtual ~NetStreamReader();
	public:
		//
		//  获取数据指针
		//
		uint8_t* getData() const;
		//
		//  获取存储的数据大小
		//
		uint32_t getSize() const;
		//
		//  获取数据的开始为ZHi
		//
		uint32_t getPosition() const;
		//
		//  丢掉这么长度的数据
		//
		void skip(int32_t length);
		//
		//  socket拉取数据填充
		//
		status_t pull(int32_t expect);
		//
		//  ByteReader 读写
		//
		virtual int8_t read1Bytes(bool* suc);
		virtual int16_t read2Bytes(bool* suc);
		virtual int32_t read3Bytes(bool* suc);
		virtual int32_t read4Bytes(bool* suc);
		virtual int64_t read8Bytes(bool* suc);
		bool readString(String& val, uint32_t len);
		bool readBytes(int8_t* data, uint32_t size);
		//
		//  跳过多少个字节
		//
		void skipRead(int32_t step);
		//
		//  是否还有这么多可以读的数据
		//
		bool haveData(uint32_t size);
	private:
		void readBuffer(uint8_t* dst, uint32_t size, bool order);
	private:
		//
		//  缓冲区，读指针，大小
		//
		ByteBuffer* mBuffer;
		uint32_t mPosition;
		uint32_t mSize;

		CSocket* mSocket;
	};
}

#endif

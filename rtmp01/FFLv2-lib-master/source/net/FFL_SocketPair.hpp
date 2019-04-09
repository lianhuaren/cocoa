/*
*  This file is part of FFL project.
*
*  The MIT License (MIT)
*  Copyright (C) 2017-2018 zhufeifei All rights reserved.
*
*  FFL_SocketPair.hpp
*  Created by zhufeifei(34008081@qq.com) on 2018/11/24
*  https://github.com/zhenfei2016/FFLv2-lib.git
*  socketpair模拟定义
*
*/

#ifndef _FFL_SOCKET_PAIR_HPP_
#define _FFL_SOCKET_PAIR_HPP_
#include <net/FFL_Net.h>
#include <FFL_Core.h>

namespace FFL{
	class FFLIB_API_IMPORT_EXPORT SocketPair{		
	public:
		SocketPair();
		~SocketPair();
		
		bool create();
		void destroy();

		NetFD getFd0() const;
		NetFD getFd1() const;

		//
		//  fd0上写
		//
		bool writeFd0(const uint8_t* data,size_t size,size_t* writedSize);
		//
		//  fd1上读
		//
		bool readFd1(uint8_t* data, size_t size, size_t* readedSize);
	private:
		DISABLE_COPY_CONSTRUCTORS(SocketPair);
	private:
		NetFD mFd[2];
	}; 
}

#endif 

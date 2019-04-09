/*
*  This file is part of FFL project.
*
*  The MIT License (MIT)
*  Copyright (C) 2017-2018 zhufeifei All rights reserved.
*
*  FFL_Flags   
*  Created by zhufeifei(34008081@qq.com) on 2018/03/06 
*  https://github.com/zhenfei2016/FFLv2-lib.git
*
标志位存取
*
*/
#ifndef _FFL_PIPELINE_FALGS_HPP_
#define _FFL_PIPELINE_FALGS_HPP_

#include <FFL_Core.h>

namespace FFL{
   class Flags32b{
   public:
	   inline Flags32b():mFlags(0)
	   {}

		//
		//   add：添加的标志
		//   remove :移除的标志
		//   注意是先添加后移除
		//
		uint32_t modifyFlags(uint32_t add, uint32_t remove);
		//
		//  是否存在一个标志
		//
		bool hasFlags(uint32_t flags) const;
		//
		//  修改状态标志
		//
		inline uint32_t resetFlags(uint32_t flag)
		{
			mFlags = flag;
			return mFlags;
		}
		//
		//  获取标志值
		//
		inline uint32_t getFlags() const 
		{
			return mFlags; 
		};
   protected:
	   uint32_t mFlags;
	};
}


#endif


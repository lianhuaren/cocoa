/*
*  This file is part of FFL project.
*
*  The MIT License (MIT)
*  Copyright (C) 2017-2018 zhufeifei All rights reserved.
*
*  FFL_Flags
*  Created by zhufeifei(34008081@qq.com) on 2018/03/06
*  https://github.com/zhenfei2016/FFLv2-lib.git
*  标志位存取
*
*/
#include <FFL_Flags.hpp>

namespace FFL {
	//
	//   add：添加的标志
	//   remove :移除的标志
	//
	uint32_t Flags32b::modifyFlags(uint32_t add, uint32_t remove)
	{
		if (add)
			mFlags |= add;

		if (remove)
			mFlags &= ~remove;
		return mFlags;
	}
	bool Flags32b::hasFlags(uint32_t flags) const {
		return (mFlags & (flags)) != 0;
	}
}
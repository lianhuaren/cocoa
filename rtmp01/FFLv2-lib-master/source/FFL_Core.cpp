/*
*  This file is part of FFL project.
*
*  The MIT License (MIT)
*  Copyright (C) 2017-2018 zhufeifei All rights reserved.
*
*  FFL_Core.cpp
*  Created by zhufeifei(34008081@qq.com) on 2019/02/01
*  https://github.com/zhenfei2016/FFLv2-lib.git
*
*  FFL 全局初始化，结束函数定义，实现
*
*/
#include <FFL_Core.h>

//
//  自动进行库的初始化
//
class AutoInitFFLib {
public:
	AutoInitFFLib() {
		FFL_initialize();
	}

	~AutoInitFFLib() {
		FFL_terminate();
	}
};
//
// 如果定义了这个宏(FFl_Config.h中通过cmake脚本生成) FFL_DISABLE_AUTO_INITLIB，则关闭自动初始化功能
//
#ifdef FFL_DISABLE_AUTO_INITLIB
static AutoInitFFLib gInitFFLib;
#endif 


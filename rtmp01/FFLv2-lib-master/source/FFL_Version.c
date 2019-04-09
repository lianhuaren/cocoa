/*
*  This file is part of FFL project.
*
*  The MIT License (MIT)
*  Copyright (C) 2017-2018 zhufeifei All rights reserved.
*
*  FFL_Version.c
*  Created by zhufeifei(34008081@qq.com) on 2018/1/7.
*  https://github.com/zhenfei2016/FFLv2-lib.git
*  版本相关信息
*
*/

#include <FFL_Version.h>


static const char* gFFLVersion = FFLLIB_VERSION_STRING;
/*
 *  获取版本号，字符串类型
 */
const char* FFL_CALL FFL_GetVersion(){
	return gFFLVersion;
}

/*
 *  获取版本号，整形
 */
int FFL_CALL FFL_GetVersion_int(){
	return FFLLIB_VERSION_INT;
}

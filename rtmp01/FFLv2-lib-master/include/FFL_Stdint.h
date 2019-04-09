/*
*  This file is part of FFL project.
*
*  The MIT License (MIT)
*  Copyright (C) 2017-2018 zhufeifei All rights reserved.
*
*  FFL_Stdint.h   
*  Created by zhufeifei(34008081@qq.com) on 2019/02/1 
*  https://github.com/zhenfei2016/FFLv2-lib.git
*
*
*/
#ifndef _FFL_STDINT_H_
#define _FFL_STDINT_H_

/*  支持  int8_t * 系列的定义 */
#ifdef WIN32
   /*  c99支持的 */
  #include <stdint.h>
#else
  #include <stdint.h>
  #include <stddef.h>
#endif

#endif

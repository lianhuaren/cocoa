/*
*  This file is part of FFL project.
*
*  The MIT License (MIT)
*  Copyright (C) 2017-2018 zhufeifei All rights reserved.
*
*  FFL_Memory.c
*  Created by zhufeifei(34008081@qq.com) on 2017/7/12.
*  https://github.com/zhenfei2016/FFLv2-lib.git
*
*  内存的申请，释放，大小端控制 ， 内存泄漏检测
*/

#include "FFL_Memory.h"
#if  CHECK_FOR_MEMORY_LEAKS
#include "memoryLeak.c"
#else
#include "memory.c"
#endif

/*
*  内存申请，并且清空
*/
void *FFL_mallocz(size_t size) {
	void *mem = FFL_malloc(size);
	if (mem)
		memset(mem, 0, size);
	return mem;
}

int FFL_isLittleEndian(){
	static int littleEndian = -1;
	if (littleEndian == -1) {
		union {
			int32_t i;
			int8_t c;
		} check;

		check.i = 0x01;
		littleEndian = check.c;
	}
	return littleEndian;
}

/*
   order:顺序的，还是反序的
*/
 void FFL_copyBytes(uint8_t* s, uint8_t* d, uint32_t size, int order) {
	if (order) {
		memcpy(d, s, size);
	}
	else {
		for (uint32_t i = 0; i < size; i++) {
			d[size - i - 1] = s[i];
		}
	}
}
/* 
 *  This file is part of FFL project.
 *
 *  The MIT License (MIT)
 *  Copyright (C) 2017-2018 zhufeifei All rights reserved.
 *
 *  FFL_Utils.hpp 
 *  Created by zhufeifei(34008081@qq.com) on 七月 2017. 
 *  
 *  定义了一系列的公用宏，和头文件引用
 *
*/
#ifndef _FFL_UTILITY_HPP_
#define _FFL_UTILITY_HPP_

//
//   删除指针，指针数组
//
#define FFL_SafeFree(p) \
    while (p) { \
        delete p;p = 0; break;\
    }

#define FFL_SafeFreeA(p) \
    while (p) { \
        delete[] p;p = 0; break;\
    }

//
//  栈退出时候，自动删除对应的对象
//
#define FFL_AutoFree(className, instance) \
internal_AutoFree<className> _auto_free_##instance(&instance, false)

template<class T>
class internal_AutoFree
{
public:
	internal_AutoFree(T** p) : ptr(p) {  }
	~internal_AutoFree() { FFL_SafeFree(*ptr); }
private:
	T** ptr;
};

//
//  栈退出时候，自动删除对应的数组对象
//
#define FFL_AutoFreeArray(className, instance) \
internal_AutoFreeA<className> _auto_free_array_##instance(&instance, true)

template<class T>
class internal_AutoFreeA
{
public:
	internal_AutoFreeA(T* * p): ptr(p){ }
	~internal_AutoFreeA() { FFL_SafeFreeA(*ptr); }
private:
	T* * ptr;
};


//
// 禁用c++的拷贝构造，赋值函数
//
#define DISABLE_COPY_CONSTRUCTORS(class_name) \
    class_name(const class_name &); \
    class_name &operator=(const class_name &)



#include <FFL_String.hpp>


#endif

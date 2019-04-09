/*
*  This file is part of FFL project.
*
*  The MIT License (MIT)
*  Copyright (C) 2017-2018 zhufeifei All rights reserved.
*
*  FFL_Serializable   
*  Created by zhufeifei(34008081@qq.com) on 2018/03/10 
*  https://github.com/zhenfei2016/FFLv2-lib.git
*
*  用于序列换的接口
*/
#ifndef _FFL_SERILIZABLE_HPP_
#define _FFL_SERILIZABLE_HPP_

#include <FFL_Ref.hpp>

namespace FFL{
   class ByteStream;
   class ISerializable{
   public:
	   virtual void serialization(ByteStream& stream)=0;
	   virtual status_t deserialization(ByteStream& stream)=0;	 
	};


   class FFLIB_API_IMPORT_EXPORT Serializable : public ISerializable, public RefBase {
   public:
	   Serializable();
	   ~Serializable();

	   //
	   //  对象序列化到stream中
	   //
	   virtual void serialization(ByteStream& stream);
	   //
	   //  stream中反序列到当前对象实例中
	   //
	   virtual status_t deserialization(ByteStream& stream);
   };
}


#endif


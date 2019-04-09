#ifndef _FFL_NET_UTILS_HPP_
#define _FFL_NET_UTILS_HPP_

#include <FFL_String.hpp>
namespace FFL {
	//
	//  分解url中的host和port
	//  host:port
	FFLIB_API_IMPORT_EXPORT void FFL_parseHostport(const String& url, String& host, int16_t& port);
	//
	//  是否一个ip, 是的话返回FFL_Ok
	//
	FFLIB_API_IMPORT_EXPORT status_t FFL_isIp(const String& ip);
}

#endif


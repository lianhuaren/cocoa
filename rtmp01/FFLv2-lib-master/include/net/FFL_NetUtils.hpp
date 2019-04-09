#ifndef _FFL_NET_UTILS_HPP_
#define _FFL_NET_UTILS_HPP_

#include <FFL_String.hpp>
namespace FFL {
	//
	//  �ֽ�url�е�host��port
	//  host:port
	FFLIB_API_IMPORT_EXPORT void FFL_parseHostport(const String& url, String& host, int16_t& port);
	//
	//  �Ƿ�һ��ip, �ǵĻ�����FFL_Ok
	//
	FFLIB_API_IMPORT_EXPORT status_t FFL_isIp(const String& ip);
}

#endif


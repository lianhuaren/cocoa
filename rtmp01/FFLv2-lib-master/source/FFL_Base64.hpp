/*

*/
#ifndef _FFL_BASE64_H_
#define _FFL_BASE64_H_

#include <FFL_Core.h>

#ifdef  __cplusplus
extern "C" {
#endif
	//
	//  编码，解码
	//
	status_t FFL_Base64Encode(const uint8_t * input, size_t len, uint8_t* output, size_t outBufSize);
	status_t FFL_Base64Decode(const uint8_t * input, size_t len, uint8_t* output, size_t outBufSize);

#ifdef  __cplusplus
}
#endif
#endif 

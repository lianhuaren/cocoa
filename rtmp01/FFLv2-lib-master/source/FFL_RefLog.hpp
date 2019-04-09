#ifndef _FFL_REF_LOG_H_
#define _FFL_REF_LOG_H_
#include "internalLogConfig.h"
#define ALOG_ASSERT(condition__,format__,__class)  if (!(condition__)) { INTERNAL_FFL_LOG_DEBUG_TAG("ref:",format__); }
#if defined(WIN32)
#define ALOGD
#else
#define ALOGD(...)  INTERNAL_FFL_LOG_DEBUG_TAG("ref:",##__VA_ARGS__)
#endif

#endif

#include <FFL_Core.h>

#if defined(MACOSX) || defined(IPHONEOS)
/* Mac OS X doesn't support sem_getvalue() as of version 10.4 */
#include "../generic/syssem.c"
#else
#include "syssem_linux.c"
#endif 

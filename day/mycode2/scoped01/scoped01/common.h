#ifndef TALK_BASE_COMMON_H_
#define TALK_BASE_COMMON_H_

#ifndef ASSERT
#define ASSERT(x) (void)0
#endif

#ifndef VERIFY
#define VERIFY(x) talk_base::ImplicitCastToBool(x)
#endif

#endif

#ifndef _SYS_MUTEXT_HPP_
#define _SYS_MUTEXT_HPP_

#include <mutex>
#include <condition_variable>


#define FFL_MUTEX_MAXWAIT   (~(uint32_t)0)

struct FFL_sys_mutex
{
	std::recursive_mutex cpp_mutex;
};


struct FFL_sys_cond
{
	std::condition_variable_any cpp_cond;
};

#endif

//
//  WorkQueue.hpp
//  pthread_queue01
//
//  Created by libb on 2021/1/21.
//

#ifndef WorkQueue_hpp
#define WorkQueue_hpp

#include <stdio.h>
#include "Function.h"

#if !defined(NO_RETURN)
#define NO_RETURN
#endif
namespace WTF {

class WorkQueue {
public:
    static void concurrentApply(size_t iterations, WTF::Function<void(size_t index)>&&);
};
} // namespace WTF

#endif /* WorkQueue_hpp */

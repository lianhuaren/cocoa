//
//  WorkQueue.cpp
//  pthread_queue01
//
//  Created by libb on 2021/1/21.
//

#include "WorkQueue.h"
//#include "Deque.h"

namespace WTF {

void WorkQueue::concurrentApply(size_t iterations, WTF::Function<void (size_t index)>&& function)
{
    if (!iterations)
        return;

    if (iterations == 1) {
        function(0);
        return;
    }

    class ThreadPool {
    public:
        ThreadPool()
        {
            // We don't need a thread for the current core.
            unsigned threadCount = 1;//numberOfProcessorCores() - 1;

//            m_workers.reserveInitialCapacity(threadCount);
//            for (unsigned i = 0; i < threadCount; ++i) {
//                m_workers.append(Thread::create("ThreadPool Worker", [this] {
//                    threadBody();
//                }));
//            }
        }

//        size_t workerCount() const { return m_workers.size(); }

        void dispatch(const WTF::Function<void ()>* function)
        {
//            LockHolder holder(m_lock);
//
//            m_queue.append(function);
//            m_condition.notifyOne();
        }

    private:
        NO_RETURN void threadBody()
        {
            while (true) {
                const WTF::Function<void ()>* function;

                {
//                    LockHolder holder(m_lock);
//
//                    m_condition.wait(m_lock, [this] {
//                        return !m_queue.isEmpty();
//                    });

//                    function = m_queue.takeFirst();
                }

                (*function)();
            }
        }

//        Lock m_lock;
//        Condition m_condition;
//        Deque<const WTF::Function<void ()>*> m_queue;

//        Vector<Ref<Thread>> m_workers;
    };
//
//    static LazyNeverDestroyed<ThreadPool> threadPool;
//    static std::once_flag onceFlag;
//    std::call_once(onceFlag, [] {
//        threadPool.construct();
//    });
//
//    // Cap the worker count to the number of iterations (excluding this thread)
//    const size_t workerCount = std::min(iterations - 1, threadPool->workerCount());
//
//    std::atomic<size_t> currentIndex(0);
//    std::atomic<size_t> activeThreads(workerCount + 1);
//
//    Condition condition;
//    Lock lock;
//
//    WTF::Function<void ()> applier = [&, function = WTFMove(function)] {
//        size_t index;
//
//        // Call the function for as long as there are iterations left.
//        while ((index = currentIndex++) < iterations)
//            function(index);
//
//        // If there are no active threads left, signal the caller.
//        if (!--activeThreads) {
//            LockHolder holder(lock);
//            condition.notifyOne();
//        }
//    };
//
//    for (size_t i = 0; i < workerCount; ++i)
//        threadPool->dispatch(&applier);
//    applier();
//
//    LockHolder holder(lock);
//    condition.wait(lock, [&] { return !activeThreads; });
}
}

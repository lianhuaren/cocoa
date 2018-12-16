//
//  SyncQueue.h
//  Boost-CMake
//
//  Created by lbb on 2018/12/16.
//

#ifndef SyncQueue_h
#define SyncQueue_h
#include <list>
#include <mutex>
#include <thread>
#include <condition_variable>
template<typename T>
class SyncQueue {
private:
    std::list<T> m_queue;
    std::mutex m_mutex;
    std::condition_variable_any m_notEmpty;
    std::condition_variable_any m_notFull;
    int m_maxSize;
    
    bool IsFull() const
    {
        return m_queue.size() == m_maxSize;
    }
    
    bool IsEmpty() const
    {
        return m_queue.empty();
    }
    
    pthread_mutex_t    mutex;
    pthread_cond_t    cond;
    
public:
    SyncQueue(int maxSize):m_maxSize(maxSize)
    {
        mutex = PTHREAD_MUTEX_INITIALIZER;
        cond = PTHREAD_COND_INITIALIZER;
    }
    
    bool Empty()
    {
        std::lock_guard<std::mutex> locker(m_mutex);
        return m_queue.empty();
    }
    
    bool Full()
    {
        std::lock_guard<std::mutex> locker(m_mutex);
        return m_queue.size() == m_maxSize;
    }
    
    void Put(const T& x)
    {
        std::lock_guard<std::mutex> locker(m_mutex);
        while (IsFull())
        {
            std::cout << "is full, wating..." << std::endl;
            m_notFull.wait(m_mutex);
        }
        m_queue.push_back(x);
        m_notEmpty.notify_one();
    }
    
    void Take(T& x)
    {
        std::lock_guard<std::mutex> locker(m_mutex);
        while (IsEmpty())
        {
            std::cout << "is empty, wating..." << std::endl;
            m_notEmpty.wait(m_mutex);
        }
        
        x = m_queue.front();
        m_queue.pop_front();
        m_notFull.notify_one();
    }
    
    void produce(const T& x)
    {
        pthread_mutex_lock(&mutex);
        if (IsFull()) {
            pthread_mutex_unlock(&mutex);
            return ;        /* array is full, we're done */
        }
        
        m_queue.push_back(x);
        
        pthread_cond_signal(&cond);
        pthread_mutex_unlock(&mutex);
    }
    
    void consume(T& x)
    {
        pthread_mutex_lock(&mutex);
        while (IsEmpty())
            pthread_cond_wait(&cond, &mutex);
        
        x = m_queue.front();
        m_queue.pop_front();
        
        pthread_mutex_unlock(&mutex);
    }
};
#endif /* SyncQueue_h */

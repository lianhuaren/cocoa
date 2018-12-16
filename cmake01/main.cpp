#include <iostream>
#include <fcntl.h>
#include <unistd.h>
#include <string>
#include <string.h>
#include <functional>
#include "Thread.h"


class EventLoopThread
{
public:
    
    EventLoopThread(std::string &name);
    ~EventLoopThread();
    EventLoopThread(const EventLoopThread&) = delete;
    EventLoopThread& operator = (const EventLoopThread) = delete;
    
    void startRun();
private:
    void threaFun();
    
//    EventLoop *loop_;
    bool exiting_;
    Thread thread_;
    
//    Mutex mutex_;
//    Condition condition_;
    
    
    bool quit_;
};

//EventLoopThread::EventLoopThread(std::string &name):
//loop_(nullptr), thread_(std::bind(&EventLoopThread::threaFun, this), name),
//mutex_(), condition_(mutex_)
EventLoopThread::EventLoopThread(std::string &name):
thread_(std::bind(&EventLoopThread::threaFun, this), name)
{
}

EventLoopThread::~EventLoopThread()
{
    exiting_ = true;
//    if(loop_ != nullptr)
//    {
//        loop_->quit();
//        thread_.join();
//    }
    thread_.join();
}


void EventLoopThread::startRun()
{
    thread_.start();
    
//    {
//        MutexGuard lock(mutex_);
//        while(loop_ == nullptr)
//        {
//            condition_.wait();
//        }
//    }
//
//    return loop_;
}

void EventLoopThread::threaFun()
{
//    EventLoop loop;
//    {
//        MutexGuard lock(mutex_);
//        loop_ = &loop;
//        condition_.signal();
//    }
//
//    loop_->run();
//    loop_= nullptr;
    quit_ = false;
    
    int i = 0;
    while (!quit_) {
        std::cout << thread_.name() << "--" << i++ << std::endl;
        
        sleep(1);
    }
}

int main(int argc, char **argv)
{
    std::string threadName = "thread01";
    EventLoopThread *pthread = new EventLoopThread(threadName);
    pthread->startRun();

    delete pthread;
    
    return 0;
}

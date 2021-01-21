//
//  Queue.cpp
//  pthread_queue01
//
//  Created by libb on 2021/1/21.
//

#include "Queue.h"

void
RTMPPacket_Free(RTMPPacket *p) {
    
}

Queue::Queue()
:_cs(CriticalSectionWrapper::CreateCriticalSection())
{
    _cond = ConditionVariableWrapper::CreateConditionVariable();
//    pthread_mutex_init(&mutexPacket, NULL);
//    pthread_cond_init(&condPacket, NULL);
}
Queue::~Queue()
{
    clean();
    delete _cs;
    if(_cond)
    {
        delete _cond;
    }
    if(_cs)
    {
        delete _cs;
    }
//    pthread_mutex_destroy(&mutexPacket);
//    pthread_cond_destroy(&condPacket);
}

int Queue::putRtmpPacket(RTMPPacket *packet)
{
//    pthread_mutex_lock(&mutexPacket);
    _cs->Enter();
    
    queuePacket.push(packet);
    
//    pthread_cond_signal(&condPacket);
//    pthread_mutex_unlock(&mutexPacket);
    _cond->Wake();
    _cs->Leave();
    
    return 0;
}

RTMPPacket *Queue::getRtmpPacket()
{
//    pthread_mutex_lock(&mutexPacket);
    _cs->Enter();
    RTMPPacket *rtmpPacket = NULL;
    if (!queuePacket.empty()) {
        rtmpPacket = queuePacket.front();
        queuePacket.pop();
    } else {
//        pthread_cond_wait(&condPacket, &mutexPacket);
        _cond->SleepCS(*_cs);
    }
//    pthread_mutex_unlock(&mutexPacket);
    _cs->Leave();
    
    return rtmpPacket;
}

void Queue::clean()
{
    notifyQueue();
//    pthread_mutex_lock(&mutexPacket);
    _cs->Enter();
    while (true) {
        if (queuePacket.empty()) {
            break;
        }
        RTMPPacket *rtmpPacket = queuePacket.front();
        queuePacket.pop();
        RTMPPacket_Free(rtmpPacket);
        rtmpPacket = NULL;
        
    }
//    pthread_mutex_unlock(&mutexPacket);
    _cs->Leave();
}

void Queue::notifyQueue()
{
//    pthread_mutex_lock(&mutexPacket);
//    pthread_cond_signal(&condPacket);
//    pthread_mutex_unlock(&mutexPacket);
    _cs->Enter();
    _cond->Wake();
    _cs->Leave();
}

//
//  Queue.hpp
//  pthread_queue01
//
//  Created by libb on 2021/1/21.
//

#ifndef Queue_hpp
#define Queue_hpp

#include <stdio.h>
#include <queue>
#include <pthread.h>
#include "webrtc/system_wrappers/interface/critical_section_wrapper.h"
#include "webrtc/system_wrappers/interface/condition_variable_wrapper.h"
using namespace webrtc;

typedef struct RTMPPacket {
//    uint8_t m_headerType;
//    uint8_t m_packetType;
//    uint8_t m_hasAbsTimestamp;    /* timestamp absolute or relative? */
//    int m_nChannel;
//    uint32_t m_nTimeStamp;    /* timestamp */
//    int32_t m_nInfoField2;    /* last 4 bytes in a long header */
//    uint32_t m_nBodySize;
//    uint32_t m_nBytesRead;
//    RTMPChunk *m_chunk;
//    char *m_body;
} RTMPPacket;
void RTMPPacket_Free(RTMPPacket *p);

class Queue {
public:
    std::queue<RTMPPacket *>queuePacket;
    
//    pthread_mutex_t mutexPacket;
//    pthread_cond_t condPacket;
    CriticalSectionWrapper* _cs;
    ConditionVariableWrapper* _cond;
    
public:
    Queue();
    ~Queue();
    
    int putRtmpPacket(RTMPPacket *packet);
    RTMPPacket *getRtmpPacket();
    
    void clean();
    
    void notifyQueue();
    
};


#endif /* Queue_hpp */

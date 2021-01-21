//
//  main.m
//  pthread_queue01
//
//  Created by libb on 2021/1/21.
//

#include <iostream>
#include "Queue.h"
#include "webrtc/system_wrappers/interface/thread_wrapper.h"
//#include "Function.h"

void *callbackPush(void *data);

class RtmpPush {

public:

    Queue *queue = NULL;
//    pthread_t push_thrad = NULL;
    ThreadWrapper* _thread;
    bool isStartPush;
    long startTime;

public:
    RtmpPush();

    ~RtmpPush();

    void init();

//    void pushSPSPPS(char *sps, int spsLen, char *pps, int ppsLen);
//
    void pushVideoData(char *data, int dataLen, bool keyFrame);
//
//    void pushAudioData(char *data, int dataLen);
//
//    void pushStop();

    static bool Run(ThreadObj obj);
    bool Process();
};

bool RtmpPush::Run(ThreadObj obj)
{
    RtmpPush* mgr =
        static_cast<RtmpPush*>(obj);
    return mgr->Process();
}

bool RtmpPush::Process()
{
    callbackPush(this);
    
    return true;
}
    
RtmpPush::RtmpPush() {
//    this->url = static_cast<char *>(malloc(512));
//    strcpy(this->url, url);
    this->queue = new Queue();

    _thread = ThreadWrapper::CreateThread(RtmpPush::Run, this,
                                          kRealtimePriority,
                                          "Thread");
}


RtmpPush::~RtmpPush() {
    if (this->queue) {
        delete (queue);
        queue = NULL;
    }
    if(_thread != NULL)
    {
        delete _thread;
    }
//    if (this->url) {
//        free(this->url);
//        this->url = NULL;
//    }
}

void RtmpPush::pushVideoData(char *data, int dataLen, bool keyFrame) {
    if (!this->queue) return;
    int bodySize = dataLen + 9;
    RTMPPacket *rtmpPacket = static_cast<RTMPPacket *>(malloc(sizeof(RTMPPacket)));

    queue->putRtmpPacket(rtmpPacket);


}
void *callbackPush(void *data) {
    RtmpPush *rtmpPush = static_cast<RtmpPush *>(data);
    rtmpPush->isStartPush = false;
    

    rtmpPush->isStartPush = true;
//    rtmpPush->startTime = RTMP_GetTime();

    while (rtmpPush->isStartPush) {
        RTMPPacket *packet = rtmpPush->queue->getRtmpPacket();
        
        if (packet) {
            std::cout << "getRtmpPacket packet\n";
        } else {
            std::cout << "getRtmpPacket packet empty\n";
        }
        if (packet) {
            // queue 缓存队列大小
//            int result = RTMP_SendPacket(rtmpPush->rtmp, packet, 1);
//            LOGD("RTMP_SendPacket result:%d", result);
            RTMPPacket_Free(packet);
            free(packet);
            packet = NULL;
        }
    }


    end:
        
//    pthread_exit(&rtmpPush->push_thrad);
    std::cout << "thread end\n";
    return 0;
}


void RtmpPush::init() {
//    pthread_create(&push_thrad, NULL, callbackPush, this);

    unsigned int id = 0;
    _thread->Start(id);
}


using namespace webrtc;

int main(int argc, const char * argv[]) {
    // insert code here...
    std::cout << "Hello, World!\n";
//    CriticalSectionWrapper* _cs(CriticalSectionWrapper::CreateCriticalSection());
    RtmpPush *rtmpPush = NULL;
    
    if (!rtmpPush)
        rtmpPush = new RtmpPush();

//    isExit = false;
    rtmpPush->init();
    
    while (true) {
        std::cout << "getchar()\n";
        getchar();
        //test
        
        rtmpPush->pushVideoData(NULL, 0, 0);
        
    }
    
//    Queue *queue = NULL;
//    queue = new Queue();
//
//    RTMPPacket *packet = queue->getRtmpPacket();
//    if (packet) {
//        // queue 缓存队列大小
////        int result = RTMP_SendPacket(rtmpPush->rtmp, packet, 1);
////        LOGD("RTMP_SendPacket result:%d", result);
//        RTMPPacket_Free(packet);
//        free(packet);
//        packet = NULL;
//    }
//
//
//
//    if (queue) {
//        delete (queue);
//        queue = NULL;
//    }
    
    return 0;
}

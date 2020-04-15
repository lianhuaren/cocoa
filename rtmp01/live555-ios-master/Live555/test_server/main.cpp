
#include "liveMedia.hh"
#include "BasicUsageEnvironment.hh"


//创建交互环境，用来打印相关信息的
UsageEnvironment* env;


// To make the second and subsequent client for each stream reuse the same
// input stream as the first client (rather than playing the file from the
// start for each client), change the following "False" to "True":
Boolean reuseFirstSource = False;


// To stream *only* MPEG-1 or 2 video "I" frames
// (e.g., to reduce network bandwidth),
// change the following "False" to "True":
Boolean iFramesOnly = False;


//打印相关信息的函数
static void announceStream(RTSPServer* rtspServer, ServerMediaSession* sms,
                           char const* streamName, char const* inputFileName); // fwd


int main(int argc, char** argv) {
    // Begin by setting up our usage environment:
    // 1.创建任务调度器,createNew其实就是创建类的实例
    TaskScheduler* scheduler = BasicTaskScheduler::createNew();
    // 2. 创建交互环境
    env = BasicUsageEnvironment::createNew(*scheduler);
    //以下为权限控制的代码，设置后没有权限的客户端无法进行连接
    UserAuthenticationDatabase* authDB = NULL;
#ifdef ACCESS_CONTROL
    // To implement client access control to the RTSP server, do the following:
    authDB = new UserAuthenticationDatabase;
    authDB->addUserRecord("username1", "password1"); // replace these with real strings
    // Repeat the above with each <username>, <password> that you wish to allow
    // access to the server.
#endif
    
    
    // 3. Create the RTSP server:此时就一直处于监听模客户端的连接
    RTSPServer* rtspServer = RTSPServer::createNew(*env, 8554, authDB);
    if (rtspServer == NULL) {
        *env << "Failed to create RTSP server: " << env->getResultMsg() << "\n";
        exit(1);
    }
    
    
    char const* descriptionString
    = "Session streamed by \"testOnDemandRTSPServer\"";
    
    
    // Set up each of the possible streams that can be served by the
    // RTSP server.  Each such stream is implemented using a
    // "ServerMediaSession" object, plus one or more
    // "ServerMediaSubsession" objects for each audio/video substream.
    
    
    // A H.264 video elementary stream:
    {
        char const* streamName = "H264unicast";//流名字，媒体名
        char const* inputFileName = "test.264";//文件名，当客户端输入的流名字为h264ESVideoTest时，实际上打开的是test.264文件
        // 4.创建媒体会话
        //当客户点播时，要输入流名字streamName，告诉RTSP服务器点播的是哪个流。
        //流名字和文件名的对应关系是通过增加子会话建立起来的(流名字streamName不是文件名inputFileName)。媒体会话对会话描述、会话持续时间、流名字等与会话有关的信息进行管理
        //第二个参数:媒体名、三:媒体信息、四:媒体描述
        ServerMediaSession* sms
        = ServerMediaSession::createNew(*env, streamName, streamName,
                                        descriptionString);
        //5.添加264子会话 这里的文件名才是真正打开文件的名字
        //reuseFirstSource:
        //这里的H264VideoFileS...类派生自FileServerMediaSubsession派生自OnDemandServerMediaSubsession
        //而OnDemandServerMediaSubsession和PassiveMediaSubsession共同派生自ServerMediaSubsession
        //关于读取文件之类都在这个类中实现的，如果要将点播改为直播就是要新建类继承此类然后添加新的方法
        sms->addSubsession(H264VideoFileServerMediaSubsession
                           ::createNew(*env, inputFileName, reuseFirstSource));
        //6.为rtspserver添加session
        rtspServer->addServerMediaSession(sms);
        //打印信息到标准输出
        announceStream(rtspServer, sms, streamName, inputFileName);
    }
    
    
    // Also, attempt to create a HTTP server for RTSP-over-HTTP tunneling.
    // Try first with the default HTTP port (80), and then with the alternative HTTP
    // port numbers (8000 and 8080).
    
    
    if (rtspServer->setUpTunnelingOverHTTP(80) || rtspServer->setUpTunnelingOverHTTP(8000) || rtspServer->setUpTunnelingOverHTTP(8080)) {
        *env << "\n(We use port " << rtspServer->httpServerPortNum() << " for optional RTSP-over-HTTP tunneling.)\n";
    } else {
        *env << "\n(RTSP-over-HTTP tunneling is not available.)\n";
    }
    //执行循环方法，来执行循环方法，对套接字的读取事件和对媒体文件的延时发送操作都在这个循环中完成。
    env->taskScheduler().doEventLoop(); // does not return
    
    
    return 0; // only to prevent compiler warning
}


static void announceStream(RTSPServer* rtspServer, ServerMediaSession* sms,
                           char const* streamName, char const* inputFileName) {
    char* url = rtspServer->rtspURL(sms);
    UsageEnvironment& env = rtspServer->envir();
    env << "\n\"" << streamName << "\" stream, from the file \""
    << inputFileName << "\"\n";
    env << "Play this stream using the URL \"" << url << "\"\n";
    delete[] url;
}


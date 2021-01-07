//
//  main.cpp
//  librtmp01
//
//  Created by lbb on 2019/8/17.
//  Copyright © 2019年 lbb. All rights reserved.
//

#include <iostream>

#include <stdio.h>
#include "RTMPStream.h"

int main(int argc,char* argv[])
{
    CRTMPStream rtmpSender;
    
//    bool bRet = rtmpSender.Connect("rtmp://192.168.1.3/live1/room1");
    //"/Users/yanjun/Desktop/lbb/day/ffmpeg/test.h264"
    rtmpSender.SendH264File("/Users/yanjun/Desktop/lbb/day/ffmpeg/test.h264");
    
    rtmpSender.Close();
}


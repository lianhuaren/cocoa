//
//  WebSocket.hpp
//  wsserver01
//
//  Created by ab on 2020/12/13.
//

#ifndef WebSocket_hpp
#define WebSocket_hpp

#include <stdio.h>
#include "wsserver.h"
#include "Utils.h"

class WebSocket
{
public:
    void ProcessData(string data);
public:
    WsHandler *ws;
};

#endif /* WebSocket_hpp */

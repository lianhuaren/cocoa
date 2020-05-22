//
//  main.cpp
//  scoped01
//
//  Created by Mac on 2018/12/13.
//  Copyright © 2018年 aaaTechnology. All rights reserved.
//

#include <iostream>
#include "scoped_ptr.h"
#include "sigslot.h"

class AsyncSocket {
public:
    AsyncSocket() {};
    virtual ~AsyncSocket(){
        std::cout << "~AsyncSocket\n";
    };
    
    void test() {
        std::cout << "test\n";
    }
};

class HttpListenServer {
public:
    HttpListenServer() {};
    virtual ~HttpListenServer(){};
    
    int Listen();
private:
    scoped_ptr<AsyncSocket> listener_;
};

int HttpListenServer::Listen()
{
    AsyncSocket* sock = new AsyncSocket();
    listener_.reset(sock);
    listener_->test();
    
    return 0;
}

class Switch
{
public:
    sigslot::signal0<> Clicked;
};

class Light : public sigslot::has_slots<>
{
public:
    void  Toggle(){
        std::cout << "Light Toggle\n";
    };
};




int main(int argc, const char * argv[]) {
    // insert code here...
//    HttpListenServer server;
//    server.Listen();
    
    
    
    Switch sw1, sw2;
    Light red, white;
    sw1.Clicked.connect(&red, &Light::Toggle);
    sw2.Clicked.connect(&white, &Light::Toggle);
    sw1.Clicked();
    sw2.Clicked();
    
    std::cout << "Hello, World!\n";
    return 0;
}

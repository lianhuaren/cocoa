//
//  ViewController.m
//  ios_socket01
//
//  Created by  on 2020/5/20.
//  Copyright © 2020 admin. All rights reserved.
//

#import "ViewController.h"
#import "iOSAsyncSocket.h"

#define ECHO_MSG     1

@interface ViewController ()

@property (nonatomic, strong) iOSAsyncSocket *serverSocket;
@property (nonatomic, strong) NSMutableSet *connectedSockets;

@property (nonatomic, assign) int serverPort;
@end

@implementation ViewController

- (void)viewDidLoad {
    [super viewDidLoad];
    // Do any additional setup after loading the view.
    
    [self start];
}

- (int)start{
//     [self configFileHandle];
    
    if (self.serverPort <= 0) {
        self.serverPort = 9999;
    }
    
    self.connectedSockets = [[NSMutableSet alloc] init];
    if(!self.serverSocket){
        self.serverSocket = [[iOSAsyncSocket alloc] initWithDelegate:self delegateQueue:dispatch_get_global_queue(0, 0)];
    }
  
    NSError *error = nil;
    BOOL result =  [self.serverSocket acceptOnPort:self.serverPort error:&error];

    if(!result){
        NSLog(@"%@",error);
    }else{
        NSLog(@"server start at port %d....",self.serverPort);

    }
    return result;
}

- (void)stop{
    self.serverSocket = nil;
}


- (void)socket:(iOSAsyncSocket *)sock didAcceptNewSocket:(iOSAsyncSocket *)newSocket;
{
    [self.connectedSockets addObject:newSocket];

    NSLog(@"[Server] didAcceptNewSocket. socket = %@",sock);
    [newSocket readDataWithTimeout:-1 tag:0];
}

- (void)socketDidDisconnect:(iOSAsyncSocket *)socket withError:(NSError *)error;
{
    NSLog(@"[Server] socketDidDisconnect. socket:%@",socket);
    [self.connectedSockets removeObject:socket];
}



- (void)socket:(iOSAsyncSocket *)sock didReadData:(NSData *)data withTag:(long)tag;
{
    NSLog(@"socket didReadData:%lu",(unsigned long)data.length);
    
    dispatch_async(dispatch_get_main_queue(), ^{
        @autoreleasepool {
        
//            [self.h264FileHandle writeData:data];
//
//            if (self.delegate && [self.delegate respondsToSelector:@selector(didReadData:withTag:)]) {
//                [self.delegate didReadData:data withTag:tag];
//            }
            
        }
    });
    
    // Echo message back to client
    [sock writeData:data withTimeout:-1 tag:ECHO_MSG];
}

- (void)socket:(iOSAsyncSocket *)sock didWriteDataWithTag:(long)tag
{
    // This method is executed on the socketQueue (not the main thread)
    
    if (tag == ECHO_MSG)
    {
        [sock readDataWithTimeout:-1 tag:0];
    }
}

@end

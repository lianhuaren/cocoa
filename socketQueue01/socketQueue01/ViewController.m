//
//  ViewController.m
//  socketQueue01
//
//  Created by Li on 2020/4/26.
//  Copyright © 2020 admin. All rights reserved.
//

#import "ViewController.h"

NSString *const GCDAsyncSocketQueueName = @"GCDAsyncSocket";
#define LogTrace()              {NSLog(@"%s",__func__);}
#define return_from_block  return

@interface ViewController ()
{
    dispatch_queue_t socketQueue;
    dispatch_source_t connectTimer;
}

@end

@implementation ViewController

- (void)viewDidLoad {
    [super viewDidLoad];
    // Do any additional setup after loading the view.
    
    socketQueue = dispatch_queue_create([GCDAsyncSocketQueueName UTF8String], NULL);
    
    [self startConnectTimeout:5];
}

- (void)startConnectTimeout:(NSTimeInterval)timeout
{
    if (timeout >= 0.0)
    {
        connectTimer = dispatch_source_create(DISPATCH_SOURCE_TYPE_TIMER, 0, 0, socketQueue);
        
        __weak ViewController *weakSelf = self;
        
        dispatch_source_set_event_handler(connectTimer, ^{ @autoreleasepool {
        #pragma clang diagnostic push
        #pragma clang diagnostic warning "-Wimplicit-retain-self"
        
            __strong ViewController *strongSelf = weakSelf;
            if (strongSelf == nil) return_from_block;
            
            [strongSelf doConnectTimeout];
            
        #pragma clang diagnostic pop
        }});
        
//        #if !OS_OBJECT_USE_OBJC
//        dispatch_source_t theConnectTimer = connectTimer;
//        dispatch_source_set_cancel_handler(connectTimer, ^{
//        #pragma clang diagnostic push
//        #pragma clang diagnostic warning "-Wimplicit-retain-self"
//            
//            LogVerbose(@"dispatch_release(connectTimer)");
//            dispatch_release(theConnectTimer);
//            
//        #pragma clang diagnostic pop
//        });
//        #endif
        
        dispatch_time_t tt = dispatch_time(DISPATCH_TIME_NOW, (int64_t)(timeout * NSEC_PER_SEC));
        dispatch_source_set_timer(connectTimer, tt, DISPATCH_TIME_FOREVER, 0);
        
        dispatch_resume(connectTimer);
    }
}

- (void)endConnectTimeout
{
    LogTrace();
    
    if (connectTimer)
    {
        dispatch_source_cancel(connectTimer);
        connectTimer = NULL;
    }
    
    // Increment stateIndex.
    // This will prevent us from processing results from any related background asynchronous operations.
    //
    // Note: This should be called from close method even if connectTimer is NULL.
    // This is because one might disconnect a socket prior to a successful connection which had no timeout.
    
//    stateIndex++;
//
//    if (connectInterface4)
//    {
//        connectInterface4 = nil;
//    }
//    if (connectInterface6)
//    {
//        connectInterface6 = nil;
//    }
}

- (void)doConnectTimeout
{
    
    LogTrace();
    
    [self endConnectTimeout];
//    [self closeWithError:[self connectTimeoutError]];
}


@end

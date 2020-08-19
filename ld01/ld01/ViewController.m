//
//  ViewController.m
//  ld01
//
//  Created by  on 2020/6/28.
//  Copyright © 2020 admin. All rights reserved.
//

#import "ViewController.h"
#import <dlfcn.h>
#include <assert.h>

@interface ViewController ()
@property (nonatomic,strong) id runtime_Player;
@end

@implementation ViewController

- (void)viewDidLoad {
    [super viewDidLoad];
    // Do any additional setup after loading the view.
//    [self runtimePlay];
    
    printf("come in\n");
     
    NSString* resourcePath = [[NSBundle mainBundle] resourcePath];
    NSString* dlPath = [NSString stringWithFormat: @"%@/libtest.so", resourcePath];
    const char* cdlpath = [dlPath UTF8String];
    
    void *lib = dlopen("libtest.so", RTLD_LAZY);
    
    void *handler = dlopen(cdlpath, RTLD_LAZY);
     printf("dlopen - %sn", dlerror());
    assert(handler != NULL);
    
    void (*pTest)(int);
    pTest = (void (*)(int))dlsym(handler, "add");
    
    (*pTest)(10);
 
    dlclose(handler);
    
    printf("go out\n");
    
    
    return;
    NSString * const* foregroundConstant = (NSString * const *) dlsym(RTLD_DEFAULT, "UIApplicationWillEnterForegroundNotification");
    NSLog(@"foregroundConstant:%@",*foregroundConstant);
    if (foregroundConstant) {
        [[NSNotificationCenter defaultCenter] addObserver:self
                                                 selector:@selector(enteringForeground)
                                                     name:*foregroundConstant
                                                   object:nil];
    }
    
}

- (void)runtimePlay{
    // 获取音乐资源路径
    NSString *path = [[NSBundle mainBundle] pathForResource:@"rain" ofType:@"mp3"];
    // 加载库到当前运行程序
    void *lib = dlopen("/System/Library/Frameworks/AVFoundation.framework/AVFoundation", RTLD_LAZY);
    if (lib) {
        // 获取类名称
        Class playerClass = NSClassFromString(@"AVAudioPlayer");
        // 获取函数方法
        SEL selector = NSSelectorFromString(@"initWithData:error:");
        // 调用实例化对象方法
        _runtime_Player = [[playerClass alloc] performSelector:selector withObject:[NSData dataWithContentsOfFile:path] withObject:nil];
        // 获取函数方法
        selector = NSSelectorFromString(@"play");
        // 调用播放方法
        [_runtime_Player performSelector:selector];
        NSLog(@"动态加载播放");
    }
}

- (void) enteringForeground {
    NSLog(@"enteringForeground");
}

- (void) dealloc {
    [[NSNotificationCenter defaultCenter] removeObserver:self];
}


@end

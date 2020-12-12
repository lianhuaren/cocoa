//
//  ViewController.m
//  curl01
//
//  Created by Mac on 2019/2/21.
//  Copyright © 2019 aaaTechnology. All rights reserved.
//

#import "ViewController.h"
#include "CFileDownLoad.h"

@interface ViewController ()

@end

@implementation ViewController

- (void)viewDidLoad {
    [super viewDidLoad];
    // Do any additional setup after loading the view, typically from a nib.
    [[NSNotificationCenter defaultCenter] removeObserver:self name:@"ANYCHATNOTIFY" object:nil];
    [[NSNotificationCenter defaultCenter] addObserver:self selector:@selector(AnyChatNotifyHandler:) name:@"ANYCHATNOTIFY" object:nil];
    

    ll::CFileDownLoad::test();
    
}

- (void)AnyChatNotifyHandler:(NSNotification*)notify {
    NSDictionary* dict = notify.userInfo;
    NSLog(@"%@",dict);
//    [self.anyChat OnRecvAnyChatNotify:dict];
}

@end

//
//  ViewController.m
//  frame01
//
//  Created by Mac on 2018/12/14.
//  Copyright © 2018年 BaiRuiTechnology. All rights reserved.
//

#import "ViewController.h"
#import <LLTool/LLTool.h>
//#import <AnyChatSDK/AnyChatInitOpt.h>

@interface ViewController ()

@end

@implementation ViewController

- (void)viewDidLoad {
    [super viewDidLoad];
    // Do any additional setup after loading the view, typically from a nib.
    
    LLColorView *colorView = [[LLColorView alloc] initWithFrame:CGRectZero];
    colorView.frame = CGRectMake(0, 0, 200, 200);
    colorView.backgroundColor = [UIColor redColor];
    
    [self.view addSubview:colorView];
    
}


- (void)didReceiveMemoryWarning {
    [super didReceiveMemoryWarning];
    // Dispose of any resources that can be recreated.
}


@end

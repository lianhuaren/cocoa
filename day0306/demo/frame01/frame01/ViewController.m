//
//  ViewController.m
//  frame01
//
//  Created by Mac on 2018/12/14.
//  Copyright © 2018年 aaaTechnology. All rights reserved.
//

#import "ViewController.h"
#import <LLTool/LLTool.h>
//#import <AnyChatSDK/AnyChatInitOpt.h>

@interface ViewController ()

@property (nonatomic,strong) LLSettingView *settingView;

@end

@implementation ViewController

- (void)viewDidLoad {
    [super viewDidLoad];
    // Do any additional setup after loading the view, typically from a nib.
    
//    LLColorView *colorView = [[LLColorView alloc] initWithFrame:CGRectZero];
//    colorView.frame = CGRectMake(0, 0, 200, 200);
//    colorView.backgroundColor = [UIColor redColor];
//
//    [self.view addSubview:colorView];

    //未定义requestId
    NSTimeInterval nowtime = [[NSDate date] timeIntervalSince1970]*1000;
    NSString *date = [NSString stringWithFormat:@"%llu", [[NSNumber numberWithDouble:nowtime] longLongValue]];
    NSString *part1 = @"";//2位
    NSString *part2 = @"";//3位
    if (date.length > 3) {
        part1 = [date substringToIndex:2];
        part2 = [date substringFromIndex:date.length-3];
    }
    //8位
    NSString *requestId = [NSString stringWithFormat:@"%@%@%3d",part1,part2,(arc4random() % 1000)];
    
    NSLog(@">>>>>>>>>%@",requestId);
    
    self.settingView.backgroundColor = [UIColor clearColor];
    
    
    LLSettingViewAdapter *adapter = [[LLSettingViewAdapter alloc] init];
    __weak typeof(self) weakSelf = self;
    adapter.settingButtonClickBlock = ^(UIButton *button, int tag) {
        
        [weakSelf settingBtnClick:button tag:tag];
    };
    
    
    self.settingView.adapter = adapter;
    
    
    CGRect frame = self.settingView.frame;
    frame.origin.x = SCREEN_WIDTH-AC_WIDTH_FIT(44+15);
    frame.origin.y = AC_WIDTH_FIT(135+44);
    self.settingView.frame = frame;
    
    
    [[NSOperationQueue mainQueue]addOperationWithBlock:^{
        __strong __typeof(weakSelf) strongSelf = weakSelf;
        NSLog(@">>>>>>>>>test");
    }];
}


- (void)didReceiveMemoryWarning {
    [super didReceiveMemoryWarning];
    // Dispose of any resources that can be recreated.
}

- (void)settingBtnClick:(UIButton *)button tag:(int)tag
{
    NSLog(@"settingBtnClick tag:%d",tag);
    switch (tag) {
        case LLSettingViewTag1:
        {
            
        }
            break;
        case LLSettingViewTag2:
        {
            
        }
            break;
        case LLSettingViewTag3:
        {
            
        }
            break;

        default:
            break;
    }
}

- (LLSettingView *)settingView {
    if (_settingView == nil) {
        _settingView = [[LLSettingView alloc] initWithFrame:CGRectMake(0, 0, AC_WIDTH_FIT(44), 300)];
        
        [self.view addSubview:_settingView];
    }
    return _settingView;
}


@end

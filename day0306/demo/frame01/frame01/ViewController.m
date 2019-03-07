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

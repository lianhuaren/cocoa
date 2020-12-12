//
//  LLSettingView.m
//  LLTool
//
//  Created by Mac on 2019/3/6.
//  Copyright © 2019 aaaTechnology. All rights reserved.
//

#import "LLSettingView.h"

@implementation LLSettingViewAdapter

@end

@interface LLSettingView()
@property (nonatomic, assign) int style;

@property (nonatomic, strong) NSMutableArray *buttonsArr;


@end

@implementation LLSettingView

- (instancetype)init {
    if (self = [super init]) {
        [self commonInit];
    }
    return self;
}

- (instancetype)initWithFrame:(CGRect)frame {
    if (self = [super initWithFrame:frame]) {
        [self commonInit];
    }
    return self;
}

- (id)initWithCoder:(NSCoder *)aDecoder {
    if (self = [super initWithCoder:aDecoder]) {
        [self commonInit];
    }
    return self;
}

- (void)commonInit {
    self.buttonsArr = [NSMutableArray array];
    
    
    [self addButtonsRemoveTag:LLSettingViewTagNone];
}

- (void)addButtonsRemoveTag:(int)removeTag
{
    for (UIButton *button in self.buttonsArr) {
        [button removeFromSuperview];
    }
    [self.buttonsArr removeAllObjects];
    
    [self addButtonWithTag:LLSettingViewTag1 normalImageName:nil selectedImageName:nil];
    [self addButtonWithTag:LLSettingViewTag2 normalImageName:nil selectedImageName:nil];
    [self addButtonWithTag:LLSettingViewTag3 normalImageName:nil selectedImageName:nil];

    
    CGRect frame = self.frame;
    frame.size.height = self.buttonsArr.count * AC_WIDTH_FIT(44+10) + AC_WIDTH_FIT(10);
    self.frame = frame;
}

- (void)addButtonWithTag:(int)tag normalImageName:(NSString *)normalImageName selectedImageName:(NSString *)selectedImageName
{
    UIButton *iconBtn = [UIButton buttonWithTitle:@"" atTitleColor:[UIColor whiteColor] atTarget:self atAction:@selector(iconBtnClick:)];
    [iconBtn setBackgroundColor:[UIColor blueColor]];
    
    if (normalImageName.length > 0) {
        [iconBtn setImage:[UIImage imageNamed:normalImageName] forState:UIControlStateNormal];
    }
    if (selectedImageName.length > 0) {
        [iconBtn setImage:[UIImage imageNamed:selectedImageName] forState:UIControlStateSelected];
    }
    //>>>>>>test
    {
        switch (tag) {
            case LLSettingViewTag1:
            {
                [iconBtn setTitle:@"tag1" forState:UIControlStateNormal];
            }
                break;
            case LLSettingViewTag2:
            {
                [iconBtn setTitle:@"tag2" forState:UIControlStateNormal];
            }
                break;
            case LLSettingViewTag3:
            {
                [iconBtn setTitle:@"tag3" forState:UIControlStateNormal];
            }
                break;

            default:
                break;
        }
    }
    
    [self addSubview:iconBtn];
    
    iconBtn.tag = tag;
    CGFloat y = self.buttonsArr.count * AC_WIDTH_FIT(44+10);
    iconBtn.frame = CGRectMake(0, y, AC_WIDTH_FIT(44), AC_WIDTH_FIT(44));
    
    [self.buttonsArr addObject:iconBtn];
    
}

- (void)iconBtnClick:(UIButton *)sender {
    int tag = (int)sender.tag;
    if (self.adapter.settingButtonClickBlock) {
        self.adapter.settingButtonClickBlock(sender, tag);
    }
    
//    switch (style) {
//        case LLSettingViewStyleAudio:
//        case LLSettingViewStyleSingle:
//        case LLSettingViewStyleDouble:
//        {
//            if ([self isRecording]) {
//                return;
//            }
//            [self addButtonsRemoveStyle:style];
//        }
//            break;
//            
//        default:
//            break;
//    }
}

@end

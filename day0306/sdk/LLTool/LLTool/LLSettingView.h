//
//  LLSettingView.h
//  LLTool
//
//  Created by Mac on 2019/3/6.
//  Copyright © 2019 BaiRuiTechnology. All rights reserved.
//

#import <UIKit/UIKit.h>

NS_ASSUME_NONNULL_BEGIN


typedef NS_ENUM(NSInteger, LLSettingViewTag) {
    LLSettingViewTagNone=0,  //无
    LLSettingViewTag1,  //测试1
    LLSettingViewTag2,  //测试2
    LLSettingViewTag3  //测试3

};

@interface LLSettingViewAdapter : NSObject

@property (nonatomic,copy) void(^settingButtonClickBlock)(UIButton *button, int tag);


@end


@interface LLSettingView : UIView

@property (nonatomic, strong) LLSettingViewAdapter *adapter;

@end

NS_ASSUME_NONNULL_END

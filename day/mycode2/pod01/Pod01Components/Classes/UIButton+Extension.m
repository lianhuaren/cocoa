//
//  UIButton+Extension.m
//  text
//
//  Created by 格式化油条 on 15/9/10.
//  Copyright (c) 2015年 XQBoy. All rights reserved.
//

#import "UIButton+Extension.h"
#import "Macro.h"

@implementation UIButton (Extension)

/** 创建按钮，设置按钮文字，文字颜色默认灰色，文字大小默认15 */
+ (instancetype)buttonWithTitle:(NSString *)title atTarget:(id)target atAction:(SEL)action {
    return [self buttonWithTitle:title atTitleSize:0 atTitleColor:nil atTarget:target atAction:action];
}

/** 创建按钮，设置按钮文字与大小，文字颜色默认灰色 */
+ (instancetype)buttonWithTitle:(NSString *)title atTitleSize:(CGFloat)size atTarget:(id)target atAction:(SEL)action {
    return [self buttonWithTitle:title atTitleSize:size atTitleColor:nil atTarget:target atAction:action];
}

/** 创建按钮，设置按钮文字与文字颜色，文字大小默认15 */
+ (instancetype)buttonWithTitle:(NSString *)title atTitleColor:(UIColor *)color atTarget:(id)target atAction:(SEL)action {
    return [self buttonWithTitle:title atTitleSize:0 atTitleColor:color atTarget:target atAction:action];
}

/** 创建按钮，设置按钮文字、文字颜色与文字大小 */
+ (instancetype)buttonWithTitle:(NSString *)title atTitleSize:(CGFloat)size atTitleColor:(UIColor *)color atTarget:(id)target atAction:(SEL)action {
    UIButton *button = [UIButton buttonWithType:UIButtonTypeCustom];
    [button setTitle:title forState:UIControlStateNormal];
    [button.titleLabel setFont:[UIFont systemFontOfSize: (size ? : 15 )]];
    [button setTitleColor:(color ? : [UIColor grayColor]) forState:UIControlStateNormal];
    [button addTarget:target action:action forControlEvents:UIControlEventTouchDown];
    return button;
}

/** 创建带有图片与文字的按钮 */
+ (instancetype)buttonWithTitle:(NSString *)title atNormalImageName:(NSString *)normalImageName atSelectedImageName:(NSString *)selectedImageName atTarget:(id)target atAction:(SEL)action {
    
    UIButton *button = [UIButton buttonWithType:UIButtonTypeCustom];
    [button setTitle:title forState:UIControlStateNormal];
    [button setTitleColor:RGB(192, 192, 192) forState:UIControlStateNormal];
    [button setImage:(normalImageName ? [UIImage imageNamed:normalImageName] : nil) forState:UIControlStateNormal];
    [button setImage:(selectedImageName ? [UIImage imageNamed:selectedImageName] : nil) forState:UIControlStateSelected];
    button.titleEdgeInsets = UIEdgeInsetsMake(0, 10, 0, 0);
    [button.titleLabel setFont:FONTS(15)];
    [button addTarget:target action:action forControlEvents:UIControlEventTouchUpInside];
    return button;
}

/** 创建带有图片与文字的按钮 */
+ (instancetype)buttonWithTitle:(NSString *)title atBackgroundNormalImageName:(NSString *)BackgroundImageName atBackgroundSelectedImageName:(NSString *)BackgroundselectedImageName atTarget:(id)target atAction:(SEL)action {
    
    UIButton *button = [UIButton buttonWithType:UIButtonTypeCustom];
    /** 设置标题 */
    [button setTitle:title forState:UIControlStateNormal];
    /** 设置字体颜色 */
    [button setTitleColor:RGB(192, 192, 192) forState:UIControlStateNormal];
    /** 普通背景图 */
    [button setBackgroundImage:(BackgroundImageName ? [UIImage imageNamed:BackgroundImageName] : nil) forState:UIControlStateNormal];
    /** 选中背景图 */
    [button setBackgroundImage:(BackgroundselectedImageName ? [UIImage imageNamed:BackgroundselectedImageName] : nil) forState:UIControlStateSelected];
    /** 添加点击事件 */
    [button addTarget:target action:action forControlEvents:UIControlEventTouchUpInside];
    /** 设置字体 */
    [button.titleLabel setFont:FONTS(15)];
    return button;
}

@end

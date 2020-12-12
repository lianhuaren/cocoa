//
//  UIButton+Extension.h
//  text
//
//  Created by 格式化油条 on 15/9/10.
//  Copyright (c) 2015年 XQBoy. All rights reserved.
//

#import <UIKit/UIKit.h>

@interface UIButton (Extension)

/** 创建按钮，设置按钮文字，文字颜色默认灰色，文字大小默认15 */
+ (instancetype)buttonWithTitle:(NSString *)title atTarget:(id)target atAction:(SEL)action;
/** 创建按钮，设置按钮文字与大小，文字颜色默认灰色 */
+ (instancetype)buttonWithTitle:(NSString *)title atTitleSize:(CGFloat)size atTarget:(id)target atAction:(SEL)action;
/** 创建按钮，设置按钮文字与文字颜色，文字大小默认15 */
+ (instancetype)buttonWithTitle:(NSString *)title atTitleColor:(UIColor *)color atTarget:(id)target atAction:(SEL)action;
/** 创建按钮，设置按钮文字、文字颜色与文字大小 */
+ (instancetype)buttonWithTitle:(NSString *)title atTitleSize:(CGFloat)size atTitleColor:(UIColor *)color atTarget:(id)target atAction:(SEL)action;
/** 创建带有图片与文字的按钮，文字颜色默认为灰色 */
+ (instancetype)buttonWithTitle:(NSString *)title atNormalImageName:(NSString *)normalImageName atSelectedImageName:(NSString *)selectedImageName atTarget:(id)target atAction:(SEL)action;
/** 创建带有背景图片与文字的按钮，文字颜色默认为灰色 */
+ (instancetype)buttonWithTitle:(NSString *)title atBackgroundNormalImageName:(NSString *)BackgroundImageName atBackgroundSelectedImageName:(NSString *)BackgroundselectedImageName atTarget:(id)target atAction:(SEL)action;
@end

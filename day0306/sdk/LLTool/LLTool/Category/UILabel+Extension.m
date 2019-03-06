//
//  UILabel+Extension.m
//  text
//
//  Created by 格式化油条 on 15/9/10.
//  Copyright (c) 2015年 XQBoy. All rights reserved.
//

#import "UILabel+Extension.h"

@implementation UILabel (Extension)
/** 创建label，默认文字颜色为灰色，文字大小为14 */
+ (instancetype)labelWithText:(NSString *)text {
    return [self labelWithText:text atColor:nil];
}

/** 创建label，自定义文字颜色，默认文字大小为14 */
+ (instancetype)labelWithText:(NSString *)text atColor:(UIColor *)color {
    return [self labelWithText:text atColor:color atTextSize:0];
}

/** 自定义文字大小与颜色 */
+ (instancetype)labelWithText:(NSString *)text atColor:(UIColor *)color atTextSize:(CGFloat)size {
    UILabel *label = [[self alloc] init];
    [label setText:text];
    [label setTextColor:color ? : [UIColor blackColor]];
    [label setFont:[UIFont systemFontOfSize:size ? : 14]];
    return label;
}

@end

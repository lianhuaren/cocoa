//
//  UILabel+Extension.h
//  text
//
//  Created by 格式化油条 on 15/9/10.
//  Copyright (c) 2015年 XQBoy. All rights reserved.
//

#import <UIKit/UIKit.h>

@interface UILabel (Extension)

/** 创建label，默认文字颜色为灰色，文字大小为15 */
+ (instancetype)labelWithText:(NSString *)text;
/** 创建label，自定义文字颜色，默认文字大小为15 */
+ (instancetype)labelWithText:(NSString *)text atColor:(UIColor *)color;
/** 自定义文字大小与颜色 */
+ (instancetype)labelWithText:(NSString *)text atColor:(UIColor *)color atTextSize:(CGFloat)size;

@end

//
//  ACLabel.h
//  CATextLayerTest
//
//  Created by ab on 2019/1/25.
//  Copyright © 2019年 Shang Chuanren. All rights reserved.
//

#import <UIKit/UIKit.h>

@interface ACLabel : UIView

//@property (nonatomic, assign)  CGFloat fontSize;
//@property (nonatomic, copy)  NSString *fontName;
//@property (nonatomic, copy)  NSString *text;
@property (nonatomic, copy)  NSAttributedString *attributedText;
@property (nonatomic, strong) UIFont *font;

+(instancetype)instanceWithFrame:(CGRect)frame text:(NSString *)text font:(UIFont *)font  ;
- (void)setupLayers;

@property (nonatomic, assign) CGFloat progress;

@end

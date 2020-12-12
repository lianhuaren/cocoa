//
//  ACTextScollView.h
//  CATextLayerTest
//
//  Created by ab on 2019/1/26.
//  Copyright © 2019年 Shang Chuanren. All rights reserved.
//

#import <UIKit/UIKit.h>

@interface ACTextScollView : UIScrollView

- (void)setupText:(NSString *)text font:(UIFont *)font;
- (void)setProgress:(CGFloat)progress;

@end

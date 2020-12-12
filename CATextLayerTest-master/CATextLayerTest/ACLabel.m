//
//  ACLabel.m
//  CATextLayerTest
//
//  Created by ab on 2019/1/25.
//  Copyright © 2019年 Shang Chuanren. All rights reserved.
//

#import "ACLabel.h"
@interface ACLabel()

@property (nonatomic, strong) CATextLayer *maskTextLayer;
@property (nonatomic, strong) CALayer *topLayer;
@property (nonatomic, strong) CALayer *bottomLayer;

@end

@implementation ACLabel

+(instancetype)instanceWithFrame:(CGRect)frame text:(NSString *)text font:(UIFont *)font {
    CGFloat width = frame.size.width;
    
    NSMutableParagraphStyle *paragraphStyle = [[NSMutableParagraphStyle alloc] init];
    paragraphStyle.lineSpacing = 15;// 字体的行间距
    paragraphStyle.alignment = NSTextAlignmentCenter;
    
    NSDictionary *attributes = @{
                                 NSFontAttributeName:font,
                                 NSParagraphStyleAttributeName:paragraphStyle
                                 };
    NSAttributedString *attributedText = [[NSAttributedString alloc] initWithString:text attributes:attributes];
    
    CGSize realSize = [text boundingRectWithSize:CGSizeMake(width, CGFLOAT_MAX) options:NSStringDrawingUsesLineFragmentOrigin attributes:@{NSFontAttributeName:font,NSParagraphStyleAttributeName:paragraphStyle} context:nil].size;
    frame.size.width = width;//realSize.width;
    frame.size.height = realSize.height;
    
    
    ACLabel *label = [[ACLabel alloc] initWithFrame:frame];
    label.attributedText = attributedText;
    label.font = font;
    [label setupLayers];
    
    return label;
}

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
    
    
    if (!self.backgroundColor) {
        self.backgroundColor = [UIColor whiteColor];
    }
    
    //[self setupLayers];
}

- (void)setupLayers {
    
    self.bottomLayer.mask = self.maskTextLayer;
    [self.bottomLayer addSublayer:self.topLayer];
    [self.layer addSublayer:self.bottomLayer];
    

//    CABasicAnimation *scaleAnimation = [CABasicAnimation
//                                        animationWithKeyPath:@"fontSize"];
//    scaleAnimation.duration = kAnimationDuration;
//    scaleAnimation.fillMode = kCAFillModeForwards;
//    scaleAnimation.removedOnCompletion = NO;
//    scaleAnimation.fromValue = @(kTextLayerFontSize);
//    scaleAnimation.toValue = @(kTextLayerSelectedFontSize);
//    scaleAnimation.timingFunction = [CAMediaTimingFunction
//                                     functionWithName:kCAMediaTimingFunctionLinear];

    
    self.maskTextLayer.speed = 0.0f;
    //    [self.maskTextLayer addAnimation:scaleAnimation forKey:@"animateFontSize"];
}

#pragma mark - Getters

- (CATextLayer *)maskTextLayer {
    if (!_maskTextLayer) {
        _maskTextLayer = [CATextLayer layer];
        _maskTextLayer.wrapped = YES;
        _maskTextLayer.string = self.attributedText;
        _maskTextLayer.foregroundColor = [UIColor whiteColor].CGColor;
        if (self.font) {
            // 字体名称、大小
            CFStringRef fontName = (__bridge CFStringRef)self.font.fontName;
            CGFontRef fontRef =CGFontCreateWithFontName(fontName);
            _maskTextLayer.font = fontRef;
            _maskTextLayer.fontSize = 65;//self.font.pointSize;
            CGFontRelease(fontRef);
        }
        
        
        
        _maskTextLayer.contentsScale = [[UIScreen mainScreen] scale];
        _maskTextLayer.alignmentMode = kCAAlignmentCenter;
        CGSize size = self.frame.size;
        _maskTextLayer.frame = CGRectMake(0, 0, size.width, size.height);
    }
    return _maskTextLayer;
}

- (CALayer *)topLayer {
    if (!_topLayer) {
        _topLayer = [CALayer layer];
        _topLayer.backgroundColor = [UIColor redColor].CGColor;
        CGSize size = self.frame.size;
        _topLayer.frame = CGRectMake(0, 0, 0, size.height);
    }
    return _topLayer;
}

- (CALayer *)bottomLayer {
    if (!_bottomLayer) {
        _bottomLayer = [CALayer layer];
        _bottomLayer.backgroundColor = [UIColor blackColor].CGColor;
        CGSize size = self.frame.size;
        _bottomLayer.frame = CGRectMake(0, 0, size.width, size.height);
    }
    return _bottomLayer;
}

- (void)setProgress:(CGFloat)progress {
    if (progress > 1) {
        progress = 1;
    }
    _progress = progress;
    
    [CATransaction begin];
    [CATransaction setDisableActions:YES];
    CGRect rect = self.topLayer.frame;
    rect.size.width = roundf(self.bottomLayer.frame.size.width * progress);
    self.topLayer.frame = rect;
    [CATransaction commit];
}



@end

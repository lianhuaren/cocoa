//
//  ViewController.m
//  CATextLayerTest
//
//  Created by Joe Shang on 7/14/15.
//  Copyright (c) 2015 Shang Chuanren. All rights reserved.
//

#import "ViewController.h"
#import "ACLabel.h"
#import "ACTextScollView.h"

#define AC_PLUS_HEIGHT 736.0

#define AC_PLUS_WIDTH  414.0

#define AC_WIDTH_FIT(width) (((width)/AC_PLUS_WIDTH)*320)

#define AC_HEIGHT_FIT(height) (((height)/AC_PLUS_HEIGHT)*568)

#define SCREEN_WIDTH ([UIScreen mainScreen].bounds.size.width)

#define SCREEN_HEIGHT ([UIScreen mainScreen].bounds.size.height)

static CGFloat const kTextLayerFontSize = 29.0f;
static CGFloat const kTextLayerSelectedFontSize = 35.0f;
static CGFloat const kLayerWidth = 300.0f;
static CGFloat const kLayerHeight = 40.0f;
static CGFloat const kAnimationDuration = 5.0f;



@interface ViewController ()

@property (nonatomic, strong) CATextLayer *colorTextLayer;
@property (nonatomic, strong) CATextLayer *maskTextLayer;
@property (nonatomic, strong) CALayer *topLayer;
@property (nonatomic, strong) CALayer *bottomLayer;


//@property (strong, nonatomic)  ACLabel *yalLabel;
//@property (nonatomic, strong) NSMutableArray *labelArr;
//@property (nonatomic, strong) NSArray *lineArr;
//@property (nonatomic, assign) NSUInteger lineTotalCount;
//@property (nonatomic, assign) NSUInteger currentLabelIndex;

@property (nonatomic, strong) ACTextScollView *readTextScrollView;

@end

@implementation ViewController

#pragma mark - Life Cycle

- (ACTextScollView *)readTextScrollView {
    if (_readTextScrollView == nil) {
        
        _readTextScrollView = [[ACTextScollView alloc] init];
        _readTextScrollView.showsVerticalScrollIndicator = NO;
        //_readTextScrollView.backgroundColor = [UIColor redColor];
        [self.view addSubview:_readTextScrollView];
    }
    return _readTextScrollView;
}

- (void)viewDidLoad {
    [super viewDidLoad];
    
    //[self setupLayers];
    
    UIView *testView = [[UIView alloc] initWithFrame:CGRectMake(0, 0, 100, 100)];
    testView.backgroundColor = [UIColor yellowColor];
    [self.view addSubview:testView];
    
    self.readTextScrollView.frame = CGRectMake((SCREEN_WIDTH-320)/2, 100, 320, 142);
    
    NSString *readText = @"我本次购买xxx\n的资金为自有资金。\n我已阅读过该产品的\n推介材料和相关合同。\n清楚了解风险和收益情况，\n对本产品不承诺保本和\n最低收益表示知晓和认可。";
    
    [self.readTextScrollView setupText:readText font:[UIFont systemFontOfSize:(25)]];
//    {
//        NSArray *strArr = [readText componentsSeparatedByString:@"\n"];//@[@"我本次购买xxx",@"的资金为自有资金。"];
//
//        {
//
//            NSMutableArray *countArr = [NSMutableArray arrayWithCapacity:strArr.count+1];
//            NSUInteger count = 0;
//
//            for (int i=0; i<strArr.count; i++) {
//                NSString *text = [strArr objectAtIndex:i];
//                count += text.length;
//
//                [countArr addObject:@(count)];
//                count += 1;//换行停顿
//            }
//
//            [countArr insertObject:@(0) atIndex:0];
//            self.lineTotalCount = count;
//
//
//            NSMutableArray *lineArr = [NSMutableArray arrayWithCapacity:strArr.count];
//            for (int i=0; i<strArr.count; i++) {
//                ACLineData *lineData = [[ACLineData alloc] init];
//                lineData.text = [strArr objectAtIndex:i];
//                lineData.begin = [[countArr objectAtIndex:i] unsignedIntegerValue];
//
//                [lineArr addObject:lineData];
//            }
//            self.lineArr = lineArr;
//
//
//        }
//        CGFloat x = 0;
//        CGFloat y = 0;
//        self.readTextScrollView.frame = CGRectMake((SCREEN_WIDTH-320)/2, 100, 320, 142);
//
//        self.labelArr = [NSMutableArray array];
//        for (int i=0; i<strArr.count; i++) {
//
//            NSString *text = [strArr objectAtIndex:i];//@"我会区域变色哦";//[NSString stringWithFormat:@"%@\n%@",@"我会区域变色哦",@"我会区域变色哦我会区域变色哦我会区域变色哦我会区域变色哦"];//
//            CGSize size = CGSizeMake(320, 50);
//            UIFont *font = [UIFont systemFontOfSize:(25)];
//
//            ACLabel *yalLabel = [ACLabel instanceWithFrame:CGRectMake(x, y, size.width
//                                                                      , size.height)
//                                                      text:text
//                                                      font:font];
//            [self.readTextScrollView addSubview:yalLabel];
////            CGPoint point = yalLabel.center;
////            point.x = self.readTextScrollView.center.x;
////            yalLabel.center = point;
//
//            [self.labelArr addObject:yalLabel];
//
//            y+=size.height;
//        }
//
//        self.readTextScrollView.contentSize = CGSizeMake(320, y+40);
//
//
//
//    }
    
    
//    dispatch_after(dispatch_time(DISPATCH_TIME_NOW, (int64_t)(.7 * NSEC_PER_SEC)), dispatch_get_main_queue(), ^{
//        [self performTaskWithProgress];
//        //[self addAnimationToLetterView];
//    });

}

- (void)performTaskWithProgress {
    
    dispatch_async(dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_DEFAULT, 0), ^{
        usleep(25000);
        
        for (CGFloat i = 0.f; i < 1.f; ) {
            
            dispatch_async(dispatch_get_main_queue(), ^{
                [self setProgress:i];
            });
            
            usleep(2000);
            
            i += .0001f;
            
        }
        
        dispatch_async(dispatch_get_main_queue(), ^{

        });
    });
}

- (void)viewDidLayoutSubviews {
    [super viewDidLayoutSubviews];

    CGPoint center = CGPointMake(roundf(self.view.bounds.size.width / 2), roundf(self.view.bounds.size.height / 4));
    self.colorTextLayer.position = center;
    
    center.y += roundf(self.view.bounds.size.height / 2);
    self.bottomLayer.position = center;
}

#pragma mark - Setup Method

- (void)setupLayers {
    [self.view.layer addSublayer:self.colorTextLayer];

    self.bottomLayer.mask = self.maskTextLayer;
    [self.bottomLayer addSublayer:self.topLayer];
    [self.view.layer addSublayer:self.bottomLayer];
    
    CABasicAnimation *colorAnimation = [CABasicAnimation
                                        animationWithKeyPath:@"foregroundColor"];
    colorAnimation.duration = kAnimationDuration;
    colorAnimation.fillMode = kCAFillModeForwards;
    colorAnimation.removedOnCompletion = NO;
    colorAnimation.fromValue = (id)[UIColor blackColor].CGColor;
    colorAnimation.toValue = (id)[UIColor redColor].CGColor;
    colorAnimation.timingFunction = [CAMediaTimingFunction
                                     functionWithName:kCAMediaTimingFunctionLinear];
    CABasicAnimation *scaleAnimation = [CABasicAnimation
                                        animationWithKeyPath:@"fontSize"];
    scaleAnimation.duration = kAnimationDuration;
    scaleAnimation.fillMode = kCAFillModeForwards;
    scaleAnimation.removedOnCompletion = NO;
    scaleAnimation.fromValue = @(kTextLayerFontSize);
    scaleAnimation.toValue = @(kTextLayerSelectedFontSize);
    scaleAnimation.timingFunction = [CAMediaTimingFunction
                                     functionWithName:kCAMediaTimingFunctionLinear];
    CAAnimationGroup *animationGroup = [CAAnimationGroup animation];
    animationGroup.duration = kAnimationDuration;
    animationGroup.timingFunction = [CAMediaTimingFunction
                                     functionWithName:kCAMediaTimingFunctionLinear];
    animationGroup.fillMode = kCAFillModeForwards;
    animationGroup.removedOnCompletion = NO;
    animationGroup.animations = @[colorAnimation, scaleAnimation];
    
    self.colorTextLayer.speed = 0.0f;
    [self.colorTextLayer addAnimation:animationGroup forKey:@"animateColorAndFontSize"];
    
    self.maskTextLayer.speed = 0.0f;
//    [self.maskTextLayer addAnimation:scaleAnimation forKey:@"animateFontSize"];
}

#pragma mark - Target Action

- (IBAction)onSliderValueChanged:(id)sender {
    UISlider *slider = (UISlider *)sender;

    self.colorTextLayer.timeOffset = slider.value;
    self.maskTextLayer.timeOffset = slider.value;

    [CATransaction begin];
    [CATransaction setDisableActions:YES];
    CGRect rect = self.topLayer.frame;
    rect.size.width = roundf(self.bottomLayer.frame.size.width * slider.value / kAnimationDuration);
    self.topLayer.frame = rect;
    [CATransaction commit];
    
    //self.yalLabel.progress = slider.value / kAnimationDuration;
    
    
    [self setProgress:(slider.value / kAnimationDuration)];
}


- (void)setProgress:(CGFloat)progress {
    [self.readTextScrollView setProgress:progress];
}




#pragma mark - Getters

- (CATextLayer *)colorTextLayer {
    if (!_colorTextLayer) {
        _colorTextLayer = [CATextLayer layer];
        _colorTextLayer.string = @"我会整体变色哦";
        _colorTextLayer.foregroundColor = [UIColor blackColor].CGColor;
        _colorTextLayer.fontSize = kTextLayerFontSize;
        _colorTextLayer.contentsScale = [[UIScreen mainScreen] scale];
        _colorTextLayer.alignmentMode = kCAAlignmentCenter;
        _colorTextLayer.frame = CGRectMake(0, 0, kLayerWidth, kLayerHeight);
    }
    return _colorTextLayer;
}

- (CATextLayer *)maskTextLayer {
    if (!_maskTextLayer) {
        _maskTextLayer = [CATextLayer layer];
        NSString *str = [NSString stringWithFormat:@"%@\n%@",@"我会区域变色哦",@"我会区域变色哦我会区域变色哦我会区域变色哦我会区域变色哦"];
        _maskTextLayer.string = [[NSMutableAttributedString alloc] initWithString:str];
        _maskTextLayer.foregroundColor = [UIColor whiteColor].CGColor;
        _maskTextLayer.fontSize = kTextLayerFontSize;
        _maskTextLayer.contentsScale = [[UIScreen mainScreen] scale];
        _maskTextLayer.alignmentMode = kCAAlignmentCenter;
        _maskTextLayer.frame = CGRectMake(0, 0, kLayerWidth, kLayerHeight);
    }
    return _maskTextLayer;
}

- (CALayer *)topLayer {
    if (!_topLayer) {
        _topLayer = [CALayer layer];
        _topLayer.backgroundColor = [UIColor blueColor].CGColor;
        _topLayer.frame = CGRectMake(0, 0, 0, kLayerHeight);
    }
    return _topLayer;
}

- (CALayer *)bottomLayer {
    if (!_bottomLayer) {
        _bottomLayer = [CALayer layer];
        _bottomLayer.backgroundColor = [UIColor blackColor].CGColor;
        _bottomLayer.bounds = CGRectMake(0, 0, kLayerWidth, kLayerHeight);
    }
    return _bottomLayer;
}
@end

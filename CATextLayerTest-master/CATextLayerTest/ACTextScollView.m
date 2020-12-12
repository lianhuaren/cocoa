//
//  ACTextScollView.m
//  CATextLayerTest
//
//  Created by ab on 2019/1/26.
//  Copyright © 2019年 Shang Chuanren. All rights reserved.
//

#import "ACTextScollView.h"
#import "ACLabel.h"

@interface ACLineData : NSObject
@property (nonatomic, copy) NSString *text;
@property (nonatomic, assign) NSUInteger begin;
@end

@implementation ACLineData
@end

@interface ACTextScollView()

@property (nonatomic, strong) NSMutableArray *labelArr;
@property (nonatomic, strong) NSArray *lineArr;
@property (nonatomic, assign) NSUInteger lineTotalCount;
@property (nonatomic, assign) NSUInteger currentLabelIndex;


@property (nonatomic, strong) UIFont *font;
@property (nonatomic, strong) NSString *text;

@end

@implementation ACTextScollView



- (void)setupText:(NSString *)text font:(UIFont *)font {
    self.text = text;
    self.font = font;
    
    if (self.text.length == 0) {
        return;
    }
    NSArray *strArr = [self.text componentsSeparatedByString:@"\n"];
    
    {
        
        NSMutableArray *countArr = [NSMutableArray arrayWithCapacity:strArr.count+1];
        NSUInteger count = 0;
        
        for (int i=0; i<strArr.count; i++) {
            NSString *text = [strArr objectAtIndex:i];
            count += text.length;
            
            [countArr addObject:@(count)];
            count += 1;//换行停顿
        }
        
        [countArr insertObject:@(0) atIndex:0];
        self.lineTotalCount = count;
        
        
        NSMutableArray *lineArr = [NSMutableArray arrayWithCapacity:strArr.count];
        for (int i=0; i<strArr.count; i++) {
            ACLineData *lineData = [[ACLineData alloc] init];
            lineData.text = [strArr objectAtIndex:i];
            lineData.begin = [[countArr objectAtIndex:i] unsignedIntegerValue];
            
            [lineArr addObject:lineData];
        }
        self.lineArr = lineArr;
        
        
    }
    CGFloat x = 0;
    CGFloat y = 0;
    
    self.labelArr = [NSMutableArray array];
    for (int i=0; i<strArr.count; i++) {
        
        NSString *text = [strArr objectAtIndex:i];
        CGSize size = CGSizeMake(self.frame.size.width, 0);//单行
        UIFont *font = self.font;
        
        ACLabel *yalLabel = [ACLabel instanceWithFrame:CGRectMake(x, y, size.width
                                                                  , size.height)
                                                  text:text
                                                  font:font];
        
        [self addSubview:yalLabel];
        
        [self.labelArr addObject:yalLabel];
        
        y+=yalLabel.frame.size.height;
        
        yalLabel.transform = CGAffineTransformMakeScale(0.8, 0.8);
    }
    
    self.contentSize = CGSizeMake(320, y+40);
}

- (void)setProgress:(CGFloat)progress {
    //_progress = progress;
    
    NSAssert(self.labelArr.count == self.lineArr.count, @"");
    if  (self.currentLabelIndex < self.labelArr.count) {
        NSUInteger total = self.lineTotalCount;
        CGFloat current =  progress* total;
        
        
        
        if (self.currentLabelIndex+1 < self.labelArr.count) {
            ACLineData *nextLine = [self.lineArr objectAtIndex:self.currentLabelIndex +1];
            NSUInteger next = nextLine.begin;
            
            if (current > next) {
                self.currentLabelIndex++;
                
                if (self.currentLabelIndex > 1) {
                    //向上scroll
                    ACLabel *preLabel = [self.labelArr objectAtIndex:self.currentLabelIndex-1];
                    [UIView animateWithDuration:0.25
                                     animations:^{
                                         [self setContentOffset:CGPointMake(0, preLabel.frame.origin.y)];
                                     }];
                }
            }
            
        }
        
        ACLineData *currentLine = [self.lineArr objectAtIndex:self.currentLabelIndex];
        NSUInteger begin = currentLine.begin;
        
        ACLabel *yalLabel = [self.labelArr objectAtIndex:self.currentLabelIndex];
        if (!CGAffineTransformIsIdentity(yalLabel.transform)) {
            yalLabel.transform = CGAffineTransformIdentity;
        }
        
        yalLabel.progress = (current-begin)*(1.f)/currentLine.text.length;
        
        //NSLog(@"currentLabelIndex:%lu,progress:%.2f,current:%.2f",(unsigned long)self.currentLabelIndex,yalLabel.progress,current);
        
    }
}

@end

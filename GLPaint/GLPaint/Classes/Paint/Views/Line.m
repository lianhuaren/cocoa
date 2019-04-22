//
//  Line.m
//  GLPaint
//
//  Created by jiaguanglei on 15/12/14.
//  Copyright © 2015年 roseonly. All rights reserved.
//

#import "Line.h"

@implementation Line
@synthesize begin, end, lineColor, lineWidth;


- (instancetype)init
{
    if (self = [super init]) {
        // 设置线的默认颜色 为黑色
        [self setLineColor:[UIColor blackColor]];
    }
    return self;
}

- (void)encodeWithCoder:(NSCoder *)aCoder
{
    [aCoder encodeObject:[NSValue valueWithCGPoint:self.begin] forKey:@"begin"];
    [aCoder encodeObject:[NSValue valueWithCGPoint:self.end] forKey:@"end"];
    [aCoder encodeObject:self.lineColor forKey:@"lineColor"];
    [aCoder encodeObject:[NSNumber numberWithFloat:self.lineWidth] forKey:@"lineWidth"];
}


- (id)initWithCoder:(NSCoder *)aDecoder
{
    if (self = [super init]) {
        self.begin = [[aDecoder decodeObjectForKey:@"begin"] CGPointValue];
        self.end = [[aDecoder decodeObjectForKey:@"end"] CGPointValue];
        self.lineColor = [aDecoder decodeObjectForKey:@"lineColor"];
        self.lineWidth = [[aDecoder decodeObjectForKey:@"lineWidth"] floatValue];
    }
    return self;
}

@end

//
//  PaintView.m
//  GLPaint
//
//  Created by jiaguanglei on 15/12/14.
//  Copyright © 2015年 roseonly. All rights reserved.
//

#import "PaintView.h"
#import "Common.h"
#import "UIImage+Extention.h"


@interface PaintView ()

@property (nonatomic, assign) NSUInteger length;


@end

@implementation PaintView



- (NSMutableArray *)lines{
    if (!_lines) {
        _lines = [[NSMutableArray alloc] init];
    }
    return _lines;
}


+ (instancetype)paintView
{
    return [[self alloc] init];
}


- (instancetype)init
{
    if (self = [super init]) {
        
    }
    return self;
}




/**
 *  画线
 * 每次当屏幕需要重新显示或者刷新的时候这个方法会被调用
 */
- (void)drawRect:(CGRect)rect
{
    // 获取上下文
    CGContextRef context = UIGraphicsGetCurrentContext();
    // 设置线端点样式
    CGContextSetLineCap(context, kCGLineCapRound);

    for (Line *line in self.lines) {
        // 设置线条颜色
        [[line lineColor] set];
        // 设置线宽
        CGContextSetLineWidth(context, line.lineWidth);
        // 起点, 终点
        CGContextMoveToPoint(context, line.begin.x, line.begin.y);
        CGContextAddLineToPoint(context, line.end.x, line.end.y);
        //是否允许抗锯齿
        CGContextSetAllowsAntialiasing(UIGraphicsGetCurrentContext(), true);
        //开启抗锯齿
        CGContextSetShouldAntialias(context, true);
        // 渲染
        CGContextStrokePath(context);
    }

}


- (void)undo
{
    if ([self.undoManager canUndo]) {
        [self.undoManager undo];
    }
}

- (void)redo
{
    if ([self.undoManager canRedo]) {
        [self.undoManager redo];
    }
}



#pragma mark - touchesBegan
// 当你的手指点击到屏幕的时候这个方法会被调用
- (void)touchesBegan:(NSSet<UITouch *> *)touches withEvent:(UIEvent *)event
{

    [self.undoManager beginUndoGrouping];
    for (UITouch *touch in touches) {
        CGPoint loc = [touch locationInView:self];
        Line *newLine = [[Line alloc] init];
        [newLine setBegin:loc];
        [newLine setEnd:loc];
        [newLine setLineColor:self.paintColor];
        [newLine setLineWidth:self.lineWidth];
        self.currentLine = newLine;
        [self addLine:self.currentLine];

    }
    
}



#pragma mark - touchesMoved
// 当你的手指点击屏幕后开始在屏幕移动，它会被调用。随着手指的移动，相关的对象会秩序发送该消息
- (void)touchesMoved:(NSSet<UITouch *> *)touches withEvent:(UIEvent *)event
{
    for (UITouch *touch in touches) {
        [self.currentLine setLineColor:self.paintColor];
        [self.currentLine setLineWidth:self.lineWidth];
        // 设置终点
        CGPoint loc = [touch locationInView:self];
        [self.currentLine setEnd:loc];
        if(self.currentLine){
            //清除颜色
            if ([Common color:self.paintColor isEqualToColor:[UIColor clearColor] withTolerance:0.2]) {
                
                [self removeLineByEndPoint:loc];
                
            }else{
                
                [self addLine:self.currentLine];
            }
        }
        
        Line *newLine = [[Line alloc] init];
        [newLine setBegin:loc];
        [newLine setEnd:loc];
        [newLine setLineColor:self.paintColor];
        [newLine setLineWidth:self.lineWidth];

        self.currentLine = newLine;
    }
    [self setNeedsDisplay];
}
#pragma mark touchesEnded
//  当你的手指点击屏幕之后离开的时候，它会被调用
- (void)touchesEnded:(NSSet<UITouch *> *)touches withEvent:(UIEvent *)event
{
    [self.undoManager endUndoGrouping];
    
    UIImage *endImage = [UIImage captureWithView:(UIView *)self];
//    NSLog(@"width:%.2f height:%.2f",endImage.size.width,endImage.size.height);
    
    UIImage *processedImage = [self processFromCGImage:[endImage CGImage]];
    self.thumbImageView.image = processedImage;
    
}
#pragma mark touchesCancelled
// 取消触摸时被触发，此方法必须写，不写的话会导致NSUndoManager报错
- (void)touchesCancelled:(NSSet<UITouch *> *)touches withEvent:(UIEvent *)event
{
    [self.undoManager endUndoGrouping];
}

- (UIImage *)processFromCGImage:(CGImageRef)image{
    NSDictionary *options = [NSDictionary dictionaryWithObjectsAndKeys:
                             [NSNumber numberWithBool:YES], kCVPixelBufferCGImageCompatibilityKey,
                             [NSNumber numberWithBool:YES], kCVPixelBufferCGBitmapContextCompatibilityKey,
                             nil];
    
    CVPixelBufferRef pxbuffer = NULL;
    
    CGFloat frameWidth = CGImageGetWidth(image);
    CGFloat frameHeight = CGImageGetHeight(image);
    
    CVReturn status = CVPixelBufferCreate(kCFAllocatorDefault,
                                          frameWidth,
                                          frameHeight,
//                                          kCVPixelFormatType_32ARGB,
                                          kCVPixelFormatType_32BGRA,
                                          (__bridge CFDictionaryRef) options,
                                          &pxbuffer);
    
    NSParameterAssert(status == kCVReturnSuccess && pxbuffer != NULL);
    
    CVPixelBufferLockBaseAddress(pxbuffer, 0);
    void *pxdata = CVPixelBufferGetBaseAddress(pxbuffer);
    NSParameterAssert(pxdata != NULL);
    
    CGColorSpaceRef rgbColorSpace = CGColorSpaceCreateDeviceRGB();
    
    CGContextRef context = CGBitmapContextCreate(pxdata,
                                                 frameWidth,
                                                 frameHeight,
                                                 8,
                                                 CVPixelBufferGetBytesPerRow(pxbuffer),
                                                 rgbColorSpace,
                                                 (CGBitmapInfo)kCGImageAlphaNoneSkipFirst);
    NSParameterAssert(context);
    CGContextConcatCTM(context, CGAffineTransformIdentity);
    CGContextDrawImage(context, CGRectMake(0,
                                           0,
                                           frameWidth,
                                           frameHeight),
                       image);
    
    {
        // Get the number of bytes per row for the plane pixel buffer
        size_t bytesPerRow = CVPixelBufferGetBytesPerRowOfPlane(pxbuffer,0);
        // Get the pixel buffer width and height
        size_t width = CVPixelBufferGetWidth(pxbuffer);
//        size_t height = CVPixelBufferGetHeight(pxbuffer);
        
        uint8_t *rgbabuffer = pxdata;
        for (int y=0; y<100; y++) {
            for (int x=0; x<width;x++) {
                rgbabuffer[y*bytesPerRow+x*4+0] = 0;
                rgbabuffer[y*bytesPerRow+x*4+1] = 0;
                rgbabuffer[y*bytesPerRow+x*4+2] = 255;
                rgbabuffer[y*bytesPerRow+x*4+3] = 1;
            }
        }
        
    }
    // Create a Quartz image from the pixel data in the bitmap graphics context
    CGImageRef quartzImage = CGBitmapContextCreateImage(context);
    
    CGColorSpaceRelease(rgbColorSpace);
    CGContextRelease(context);
    
    // Create an image object from the Quartz image
    UIImage *reImage = [UIImage imageWithCGImage:quartzImage];
    
    // Release the Quartz image
    CGImageRelease(quartzImage);
    
    CVPixelBufferUnlockBaseAddress(pxbuffer, 0);
    
    CVPixelBufferRelease(pxbuffer);
    
    return reImage;
}

// 橡皮擦
- (void)removeLineByEndPoint:(CGPoint)point
{
    NSPredicate *predicate = [NSPredicate predicateWithBlock:^BOOL(id  _Nonnull evaluatedObject, NSDictionary<NSString *,id> * _Nullable bindings) {
        Line *evaluatedLine = (Line *)evaluatedObject;
        return (evaluatedLine.end.x <= point.x - 1 || evaluatedLine.end.x > point.x+1) &&
        (evaluatedLine.end.y <= point.y-1 || evaluatedLine.end.y > point.y+1);
    }];
    
    NSArray *result = [self.lines filteredArrayUsingPredicate:predicate];
    if (result && result.count > 0) {
        [self.lines removeObject:result[0]];
    }
}


#pragma mark 添加线或者点
- (void)addLine:(Line *)line
{
    // 添加线条
    [[self mutableArrayValueForKey:@"lines"] addObject:line];
    [[self.undoManager prepareWithInvocationTarget:self] removeLine:line];
    [self setNeedsDisplay];
}
#pragma mark 删除线或者点
- (void)removeLine:(Line *)line
{
    if([self.lines containsObject:line]){
        [self.lines removeObject:line];
        [[self.undoManager prepareWithInvocationTarget:self] addLine:line];
        [self setNeedsDisplay];
    }
}

- (UIImage *)paintImage
{
    return [UIImage captureWithView:(UIView *)self];
}

- (BOOL)canBecomeFirstResponder
{
    return YES;
}

- (void)didMoveToWindow
{
    [self becomeFirstResponder];
}

@end

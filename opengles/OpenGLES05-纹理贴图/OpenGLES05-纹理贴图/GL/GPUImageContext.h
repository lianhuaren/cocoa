//
//  GPUImageContext.h
//  OpenGLES02-着色器
//
//  Created by libb on 2020/12/28.
//  Copyright © 2020 qinmin. All rights reserved.
//

#import <Foundation/Foundation.h>
#import "GLProgram.h"

#define GPUImageRotationSwapsWidthAndHeight(rotation) ((rotation) == kGPUImageRotateLeft || (rotation) == kGPUImageRotateRight || (rotation) == kGPUImageRotateRightFlipVertical || (rotation) == kGPUImageRotateRightFlipHorizontal)
typedef NS_ENUM(NSUInteger, GPUImageRotationMode) {
    kGPUImageNoRotation,
    kGPUImageRotateLeft,
    kGPUImageRotateRight,
    kGPUImageFlipVertical,
    kGPUImageFlipHorizonal,
    kGPUImageRotateRightFlipVertical,
    kGPUImageRotateRightFlipHorizontal,
    kGPUImageRotate180
};


NS_ASSUME_NONNULL_BEGIN

@interface GPUImageContext : NSObject

@property(readwrite, retain, nonatomic) GLProgram *currentShaderProgram;

+ (GPUImageContext *)sharedImageProcessingContext;
- (GLProgram *)programForVertexShaderString:(NSString *)vertexShaderString fragmentShaderString:(NSString *)fragmentShaderString;

+ (void)setActiveShaderProgram:(GLProgram *)shaderProgram;

@end

NS_ASSUME_NONNULL_END

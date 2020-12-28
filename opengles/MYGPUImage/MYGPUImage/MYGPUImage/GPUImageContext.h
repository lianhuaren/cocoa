//
//  GPUImageContext.h
//  opengles01
//
//  Created by libb on 2020/12/27.
//  Copyright © 2020 Rhythmic Fistman. All rights reserved.
//

#import <Foundation/Foundation.h>
#import <OpenGLES/EAGL.h>
#import <OpenGLES/ES2/gl.h>
#import <OpenGLES/ES2/glext.h>
//#import <QuartzCore/QuartzCore.h>
//#import <AVFoundation/AVUtilities.h>
//#import <mach/mach_time.h>
#import <GLKit/GLKit.h>
#import "GLProgram.h"
#import "GPUImageFramebufferCache.h"
#import "GPUImageOutput.h"

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

@property(readonly, nonatomic) dispatch_queue_t contextQueue;

@property(readonly, retain, nonatomic) EAGLContext *context;
@property(readonly) CVOpenGLESTextureCacheRef coreVideoTextureCache;
@property(readonly) GPUImageFramebufferCache *framebufferCache;

@property(readwrite, retain, nonatomic) GLProgram *currentShaderProgram;

+ (void *)contextKey;
+ (GPUImageContext *)sharedImageProcessingContext;
+ (dispatch_queue_t)sharedContextQueue;
+ (GPUImageFramebufferCache *)sharedFramebufferCache;
+ (void)useImageProcessingContext;
//- (void)useAsCurrentContext;

// Manage fast texture upload
+ (BOOL)supportsFastTextureUpload;
+ (BOOL)deviceSupportsRedTextures;

- (void)presentBufferForDisplay;
- (GLProgram *)programForVertexShaderString:(NSString *)vertexShaderString fragmentShaderString:(NSString *)fragmentShaderString;

+ (void)setActiveShaderProgram:(GLProgram *)shaderProgram;

@end

@protocol GPUImageInput <NSObject>
- (void)newFrameReadyAtTime:(CMTime)frameTime atIndex:(NSInteger)textureIndex;
- (void)setInputFramebuffer:(GPUImageFramebuffer *)newInputFramebuffer atIndex:(NSInteger)textureIndex;
//- (NSInteger)nextAvailableTextureIndex;
- (void)setInputSize:(CGSize)newSize atIndex:(NSInteger)textureIndex;
- (void)setInputRotation:(GPUImageRotationMode)newInputRotation atIndex:(NSInteger)textureIndex;
//- (CGSize)maximumOutputSize;
//- (void)endProcessing;
//- (BOOL)shouldIgnoreUpdatesToThisTarget;
//- (BOOL)enabled;
//- (BOOL)wantsMonochromeInput;
//- (void)setCurrentlyReceivingMonochromeInput:(BOOL)newValue;
@end

NS_ASSUME_NONNULL_END

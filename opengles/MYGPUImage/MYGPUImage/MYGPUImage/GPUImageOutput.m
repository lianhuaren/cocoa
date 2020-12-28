//
//  GPUImageOutput.m
//  opengles01
//
//  Created by libb on 2020/12/27.
//  Copyright © 2020 Rhythmic Fistman. All rights reserved.
//

#import "GPUImageOutput.h"

dispatch_queue_attr_t GPUImageDefaultQueueAttribute(void)
{
#if TARGET_OS_IPHONE
    if ([[[UIDevice currentDevice] systemVersion] compare:@"9.0" options:NSNumericSearch] != NSOrderedAscending)
    {
        return dispatch_queue_attr_make_with_qos_class(DISPATCH_QUEUE_SERIAL, QOS_CLASS_DEFAULT, 0);
    }
#endif
    return nil;
}

void runSynchronouslyOnVideoProcessingQueue(void (^block)(void))
{
    dispatch_queue_t videoProcessingQueue = [GPUImageContext sharedContextQueue];
#if !OS_OBJECT_USE_OBJC
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
    if (dispatch_get_current_queue() == videoProcessingQueue)
#pragma clang diagnostic pop
#else
    if (dispatch_get_specific([GPUImageContext contextKey]))
#endif
    {
        block();
    }else
    {
        dispatch_sync(videoProcessingQueue, block);
    }
}

void runAsynchronouslyOnVideoProcessingQueue(void (^block)(void))
{
    dispatch_queue_t videoProcessingQueue = [GPUImageContext sharedContextQueue];
    
#if !OS_OBJECT_USE_OBJC
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
    if (dispatch_get_current_queue() == videoProcessingQueue)
#pragma clang diagnostic pop
#else
    if (dispatch_get_specific([GPUImageContext contextKey]))
#endif
    {
        block();
    }else
    {
        dispatch_async(videoProcessingQueue, block);
    }
}


@implementation GPUImageOutput
@synthesize outputTextureOptions = _outputTextureOptions;

- (id)init;
{
    if (!(self = [super init]))
    {
        return nil;
    }

    targets = [[NSMutableArray alloc] init];
    targetTextureIndices = [[NSMutableArray alloc] init];
//    _enabled = YES;
//    allTargetsWantMonochromeData = YES;
//    usingNextFrameForImageCapture = NO;
    
    // set default texture options
    _outputTextureOptions.minFilter = GL_LINEAR;
    _outputTextureOptions.magFilter = GL_LINEAR;
    _outputTextureOptions.wrapS = GL_CLAMP_TO_EDGE;
    _outputTextureOptions.wrapT = GL_CLAMP_TO_EDGE;
    _outputTextureOptions.internalFormat = GL_RGBA;
    _outputTextureOptions.format = GL_BGRA;
    _outputTextureOptions.type = GL_UNSIGNED_BYTE;

    return self;
}

- (NSArray*)targets;
{
    return [NSArray arrayWithArray:targets];
}

- (void)addTarget:(id<GPUImageInput>)newTarget;
{
    NSInteger nextAvailableTextureIndex = 0;//[newTarget nextAvailableTextureIndex];
    [self addTarget:newTarget atTextureLocation:nextAvailableTextureIndex];
    
//    if ([newTarget shouldIgnoreUpdatesToThisTarget])
//    {
//        _targetToIgnoreForUpdates = newTarget;
//    }
}

- (void)addTarget:(id<GPUImageInput>)newTarget atTextureLocation:(NSInteger)textureLocation;
{
    if([targets containsObject:newTarget])
    {
        return;
    }
    
//    cachedMaximumOutputSize = CGSizeZero;
    runSynchronouslyOnVideoProcessingQueue(^{
        [self setInputFramebufferForTarget:newTarget atIndex:textureLocation];
        [targets addObject:newTarget];
        [targetTextureIndices addObject:[NSNumber numberWithInteger:textureLocation]];
        
//        allTargetsWantMonochromeData = allTargetsWantMonochromeData && [newTarget wantsMonochromeInput];
    });
}

- (void)removeTarget:(id<GPUImageInput>)targetToRemove;
{
    if(![targets containsObject:targetToRemove])
    {
        return;
    }
    
//    if (_targetToIgnoreForUpdates == targetToRemove)
//    {
//        _targetToIgnoreForUpdates = nil;
//    }
    
//    cachedMaximumOutputSize = CGSizeZero;
//
    NSInteger indexOfObject = [targets indexOfObject:targetToRemove];
    NSInteger textureIndexOfTarget = [[targetTextureIndices objectAtIndex:indexOfObject] integerValue];
//
//    runSynchronouslyOnVideoProcessingQueue(^{
//        [targetToRemove setInputSize:CGSizeZero atIndex:textureIndexOfTarget];
//        [targetToRemove setInputRotation:kGPUImageNoRotation atIndex:textureIndexOfTarget];
//
        [targetTextureIndices removeObjectAtIndex:indexOfObject];
        [targets removeObject:targetToRemove];
//        [targetToRemove endProcessing];
//    });
}

- (void)setInputFramebufferForTarget:(id<GPUImageInput>)target atIndex:(NSInteger)inputTextureIndex;
{
    [target setInputFramebuffer:[self framebufferForOutput] atIndex:inputTextureIndex];
}

- (GPUImageFramebuffer *)framebufferForOutput;
{
    return outputFramebuffer;
}


@end

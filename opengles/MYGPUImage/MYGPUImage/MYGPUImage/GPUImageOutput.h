//
//  GPUImageOutput.h
//  opengles01
//
//  Created by libb on 2020/12/27.
//  Copyright © 2020 Rhythmic Fistman. All rights reserved.
//

#import <Foundation/Foundation.h>
#import <UIKit/UIKit.h>
#import "GPUImageContext.h"
NS_ASSUME_NONNULL_BEGIN

dispatch_queue_attr_t GPUImageDefaultQueueAttribute(void);
void runSynchronouslyOnVideoProcessingQueue(void (^block)(void));
void runAsynchronouslyOnVideoProcessingQueue(void (^block)(void));

@interface GPUImageOutput : NSObject
{
    GPUImageFramebuffer *outputFramebuffer;
    
    NSMutableArray *targets, *targetTextureIndices;
}
@property(readwrite, nonatomic) GPUTextureOptions outputTextureOptions;

/** Returns an array of the current targets.
 */
- (NSArray*)targets;

/** Adds a target to receive notifications when new frames are available.
 
 The target will be asked for its next available texture.
 
 See [GPUImageInput newFrameReadyAtTime:]
 
 @param newTarget Target to be added
 */
//- (void)addTarget:(id<GPUImageInput>)newTarget;

- (void)addTarget:(id)newTarget;

@end

NS_ASSUME_NONNULL_END

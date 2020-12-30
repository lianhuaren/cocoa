//
//  GPUImageContext.m
//  OpenGLES02-着色器
//
//  Created by libb on 2020/12/28.
//  Copyright © 2020 qinmin. All rights reserved.
//

#import "GPUImageContext.h"

@interface GPUImageContext()
{
    NSMutableDictionary *shaderProgramCache;
}
@end

@implementation GPUImageContext

- (id)init;
{
    if (!(self = [super init]))
    {
        return nil;
    }

//    openGLESContextQueueKey = &openGLESContextQueueKey;
//    _contextQueue = dispatch_queue_create("com.sunsetlakesoftware.GPUImage.openGLESContextQueue", GPUImageDefaultQueueAttribute());
//
//#if OS_OBJECT_USE_OBJC
//    dispatch_queue_set_specific(_contextQueue, openGLESContextQueueKey, (__bridge void *)self, NULL);
//#endif
    shaderProgramCache = [[NSMutableDictionary alloc] init];
//    shaderProgramUsageHistory = [[NSMutableArray alloc] init];
    
    return self;
}

//+ (void *)contextKey {
//    return openGLESContextQueueKey;
//}

+ (GPUImageContext *)sharedImageProcessingContext;
{
    static dispatch_once_t pred;
    static GPUImageContext *sharedImageProcessingContext = nil;
    
    dispatch_once(&pred, ^{
        sharedImageProcessingContext = [[[self class] alloc] init];
    });
    return sharedImageProcessingContext;
}

- (GLProgram *)programForVertexShaderString:(NSString *)vertexShaderString fragmentShaderString:(NSString *)fragmentShaderString;
{
    NSString *lookupKeyForShaderProgram = [NSString stringWithFormat:@"V: %@ - F: %@", vertexShaderString, fragmentShaderString];
    GLProgram *programFromCache = [shaderProgramCache objectForKey:lookupKeyForShaderProgram];

    if (programFromCache == nil)
    {
        programFromCache = [[GLProgram alloc] initWithVertexShaderString:vertexShaderString fragmentShaderString:fragmentShaderString];
        [shaderProgramCache setObject:programFromCache forKey:lookupKeyForShaderProgram];

    }
    
    return programFromCache;
}

+ (void)setActiveShaderProgram:(GLProgram *)shaderProgram;
{
    GPUImageContext *sharedContext = [GPUImageContext sharedImageProcessingContext];
    [sharedContext setContextShaderProgram:shaderProgram];
}

- (void)setContextShaderProgram:(GLProgram *)shaderProgram;
{
//    EAGLContext *imageProcessingContext = [self context];
//    if ([EAGLContext currentContext] != imageProcessingContext)
//    {
//        [EAGLContext setCurrentContext:imageProcessingContext];
//    }
    
    if (self.currentShaderProgram != shaderProgram)
    {
        self.currentShaderProgram = shaderProgram;
        [shaderProgram use];
    }
}



@end

//
//  GPUImageContext.m
//  opengles01
//
//  Created by libb on 2020/12/27.
//  Copyright © 2020 Rhythmic Fistman. All rights reserved.
//

#import "GPUImageContext.h"


@interface GPUImageContext()
{
    NSMutableDictionary *shaderProgramCache;
}
@end
@implementation GPUImageContext
@synthesize context = _context;
@synthesize coreVideoTextureCache = _coreVideoTextureCache;
@synthesize framebufferCache = _framebufferCache;

static void *openGLESContextQueueKey;

- (id)init;
{
    if (!(self = [super init]))
    {
        return nil;
    }

    openGLESContextQueueKey = &openGLESContextQueueKey;
    _contextQueue = dispatch_queue_create("com.sunsetlakesoftware.GPUImage.openGLESContextQueue", GPUImageDefaultQueueAttribute());
    
#if OS_OBJECT_USE_OBJC
    dispatch_queue_set_specific(_contextQueue, openGLESContextQueueKey, (__bridge void *)self, NULL);
#endif
    shaderProgramCache = [[NSMutableDictionary alloc] init];
//    shaderProgramUsageHistory = [[NSMutableArray alloc] init];
    
    return self;
}

+ (void *)contextKey {
    return openGLESContextQueueKey;
}

+ (GPUImageContext *)sharedImageProcessingContext;
{
    static dispatch_once_t pred;
    static GPUImageContext *sharedImageProcessingContext = nil;
    
    dispatch_once(&pred, ^{
        sharedImageProcessingContext = [[[self class] alloc] init];
    });
    return sharedImageProcessingContext;
}

+ (dispatch_queue_t)sharedContextQueue;
{
    return [[self sharedImageProcessingContext] contextQueue];
}

+ (GPUImageFramebufferCache *)sharedFramebufferCache;
{
    return [[self sharedImageProcessingContext] framebufferCache];
}

+ (void)useImageProcessingContext;
{
    [[GPUImageContext sharedImageProcessingContext] useAsCurrentContext];
}

- (EAGLContext *)context;
{
    if (_context == nil)
    {
        _context = [self createContext];
        [EAGLContext setCurrentContext:_context];
        
        // Set up a few global settings for the image processing pipeline
        glDisable(GL_DEPTH_TEST);
    }
    
    return _context;
}

- (EAGLContext *)createContext;
{
//    EAGLContext *context = [[EAGLContext alloc] initWithAPI:kEAGLRenderingAPIOpenGLES2 sharegroup:_sharegroup];
    EAGLContext *context = [[EAGLContext alloc] initWithAPI:kEAGLRenderingAPIOpenGLES2];
    NSAssert(context != nil, @"Unable to create an OpenGL ES 2.0 context. The GPUImage framework requires OpenGL ES 2.0 support to work.");
    return context;
}


- (void)useAsCurrentContext;
{
    EAGLContext *imageProcessingContext = [self context];
    if ([EAGLContext currentContext] != imageProcessingContext)
    {
        [EAGLContext setCurrentContext:imageProcessingContext];
    }
}

+ (BOOL)supportsFastTextureUpload;
{
#if TARGET_IPHONE_SIMULATOR
    return NO;
#else
    
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wtautological-pointer-compare"
    return (CVOpenGLESTextureCacheCreate != NULL);
#pragma clang diagnostic pop

#endif
}

+ (BOOL)deviceSupportsRedTextures;
{
    static dispatch_once_t pred;
    static BOOL supportsRedTextures = NO;
    
    dispatch_once(&pred, ^{
        supportsRedTextures = [GPUImageContext deviceSupportsOpenGLESExtension:@"GL_EXT_texture_rg"];
    });
    
    return supportsRedTextures;
}

+ (BOOL)deviceSupportsOpenGLESExtension:(NSString *)extension;
{
    static dispatch_once_t pred;
    static NSArray *extensionNames = nil;

    // Cache extensions for later quick reference, since this won't change for a given device
    dispatch_once(&pred, ^{
        [GPUImageContext useImageProcessingContext];
        NSString *extensionsString = [NSString stringWithCString:(const char *)glGetString(GL_EXTENSIONS) encoding:NSASCIIStringEncoding];
        extensionNames = [extensionsString componentsSeparatedByString:@" "];
    });

    return [extensionNames containsObject:extension];
}

- (CVOpenGLESTextureCacheRef)coreVideoTextureCache;
{
    if (_coreVideoTextureCache == NULL)
    {
#if defined(__IPHONE_6_0)
        CVReturn err = CVOpenGLESTextureCacheCreate(kCFAllocatorDefault, NULL, [self context], NULL, &_coreVideoTextureCache);
#else
        CVReturn err = CVOpenGLESTextureCacheCreate(kCFAllocatorDefault, NULL, (__bridge void *)[self context], NULL, &_coreVideoTextureCache);
#endif
        
        if (err)
        {
            NSAssert(NO, @"Error at CVOpenGLESTextureCacheCreate %d", err);
        }

    }
    
    return _coreVideoTextureCache;
}
- (GLProgram *)programForVertexShaderString:(NSString *)vertexShaderString fragmentShaderString:(NSString *)fragmentShaderString;
{
    NSString *lookupKeyForShaderProgram = [NSString stringWithFormat:@"V: %@ - F: %@", vertexShaderString, fragmentShaderString];
    GLProgram *programFromCache = [shaderProgramCache objectForKey:lookupKeyForShaderProgram];

    if (programFromCache == nil)
    {
        programFromCache = [[GLProgram alloc] initWithVertexShaderString:vertexShaderString fragmentShaderString:fragmentShaderString];
        [shaderProgramCache setObject:programFromCache forKey:lookupKeyForShaderProgram];
//        [shaderProgramUsageHistory addObject:lookupKeyForShaderProgram];
//        if ([shaderProgramUsageHistory count] >= MAXSHADERPROGRAMSALLOWEDINCACHE)
//        {
//            for (NSUInteger currentShaderProgramRemovedFromCache = 0; currentShaderProgramRemovedFromCache < 10; currentShaderProgramRemovedFromCache++)
//            {
//                NSString *shaderProgramToRemoveFromCache = [shaderProgramUsageHistory objectAtIndex:0];
//                [shaderProgramUsageHistory removeObjectAtIndex:0];
//                [shaderProgramCache removeObjectForKey:shaderProgramToRemoveFromCache];
//            }
//        }
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
    EAGLContext *imageProcessingContext = [self context];
    if ([EAGLContext currentContext] != imageProcessingContext)
    {
        [EAGLContext setCurrentContext:imageProcessingContext];
    }
    
    if (self.currentShaderProgram != shaderProgram)
    {
        self.currentShaderProgram = shaderProgram;
        [shaderProgram use];
    }
}

- (GPUImageFramebufferCache *)framebufferCache;
{
    if (_framebufferCache == nil)
    {
        _framebufferCache = [[GPUImageFramebufferCache alloc] init];
    }
    
    return _framebufferCache;
}

- (void)presentBufferForDisplay;
{
    [self.context presentRenderbuffer:GL_RENDERBUFFER];
}


@end

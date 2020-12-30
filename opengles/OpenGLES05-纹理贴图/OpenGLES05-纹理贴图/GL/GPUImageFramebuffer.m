//
//  GPUImageFramebuffer.m
//  OpenGLES05-纹理贴图
//
//  Created by libb on 2020/12/28.
//  Copyright © 2020 qinmin. All rights reserved.
//

#import "GPUImageFramebuffer.h"

@interface GPUImageFramebuffer()
{
    GLuint framebuffer;
//#if TARGET_IPHONE_SIMULATOR || TARGET_OS_IPHONE
//    CVPixelBufferRef renderTarget;
//    CVOpenGLESTextureRef renderTexture;
//    NSUInteger readLockCount;
//#else
//#endif
//    NSUInteger framebufferReferenceCount;
//    BOOL referenceCountingDisabled;
}



@end

@implementation GPUImageFramebuffer
@synthesize size = _size;
@synthesize textureOptions = _textureOptions;
@synthesize texture = _texture;
@synthesize missingFramebuffer = _missingFramebuffer;

- (id)initWithSize:(CGSize)framebufferSize textureOptions:(GPUTextureOptions)fboTextureOptions onlyTexture:(BOOL)onlyGenerateTexture;
{
    if (!(self = [super init]))
    {
        return nil;
    }
    
    _textureOptions = fboTextureOptions;
    _size = framebufferSize;
//    framebufferReferenceCount = 0;
//    referenceCountingDisabled = NO;
    _missingFramebuffer = onlyGenerateTexture;

//    if (_missingFramebuffer)
//    {
//        runSynchronouslyOnVideoProcessingQueue(^{
//            [GPUImageContext useImageProcessingContext];
//            [self generateTexture];
//            framebuffer = 0;
//        });
//    }
//    else
    {
        [self generateFramebuffer];
    }
    return self;
}

- (void)generateFramebuffer;
{
//    runSynchronouslyOnVideoProcessingQueue(^{
//        [GPUImageContext useImageProcessingContext];
    
        glGenFramebuffers(1, &framebuffer);
        glBindFramebuffer(GL_FRAMEBUFFER, framebuffer);
        
//        // By default, all framebuffers on iOS 5.0+ devices are backed by texture caches, using one shared cache
//        if ([GPUImageContext supportsFastTextureUpload])
//        {
//#if TARGET_IPHONE_SIMULATOR || TARGET_OS_IPHONE
//            CVOpenGLESTextureCacheRef coreVideoTextureCache = [[GPUImageContext sharedImageProcessingContext] coreVideoTextureCache];
//            // Code originally sourced from http://allmybrain.com/2011/12/08/rendering-to-a-texture-with-ios-5-texture-cache-api/
//
//            CFDictionaryRef empty; // empty value for attr value.
//            CFMutableDictionaryRef attrs;
//            empty = CFDictionaryCreate(kCFAllocatorDefault, NULL, NULL, 0, &kCFTypeDictionaryKeyCallBacks, &kCFTypeDictionaryValueCallBacks); // our empty IOSurface properties dictionary
//            attrs = CFDictionaryCreateMutable(kCFAllocatorDefault, 1, &kCFTypeDictionaryKeyCallBacks, &kCFTypeDictionaryValueCallBacks);
//            CFDictionarySetValue(attrs, kCVPixelBufferIOSurfacePropertiesKey, empty);
//
//            CVReturn err = CVPixelBufferCreate(kCFAllocatorDefault, (int)_size.width, (int)_size.height, kCVPixelFormatType_32BGRA, attrs, &renderTarget);
//            if (err)
//            {
//                NSLog(@"FBO size: %f, %f", _size.width, _size.height);
//                NSAssert(NO, @"Error at CVPixelBufferCreate %d", err);
//            }
//
//            err = CVOpenGLESTextureCacheCreateTextureFromImage (kCFAllocatorDefault, coreVideoTextureCache, renderTarget,
//                                                                NULL, // texture attributes
//                                                                GL_TEXTURE_2D,
//                                                                _textureOptions.internalFormat, // opengl format
//                                                                (int)_size.width,
//                                                                (int)_size.height,
//                                                                _textureOptions.format, // native iOS format
//                                                                _textureOptions.type,
//                                                                0,
//                                                                &renderTexture);
//            if (err)
//            {
//                NSAssert(NO, @"Error at CVOpenGLESTextureCacheCreateTextureFromImage %d", err);
//            }
//
//            CFRelease(attrs);
//            CFRelease(empty);
//
//            glBindTexture(CVOpenGLESTextureGetTarget(renderTexture), CVOpenGLESTextureGetName(renderTexture));
//            _texture = CVOpenGLESTextureGetName(renderTexture);
//            glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, _textureOptions.wrapS);
//            glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, _textureOptions.wrapT);
//
//            glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, CVOpenGLESTextureGetName(renderTexture), 0);
//#endif
//        }
//        else
        {
            [self generateTexture];

            glBindTexture(GL_TEXTURE_2D, _texture);
            
            glTexImage2D(GL_TEXTURE_2D, 0, _textureOptions.internalFormat, (int)_size.width, (int)_size.height, 0, _textureOptions.format, _textureOptions.type, 0);
            glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, _texture, 0);
        }
        
        #ifndef NS_BLOCK_ASSERTIONS
        GLenum status = glCheckFramebufferStatus(GL_FRAMEBUFFER);
        NSAssert(status == GL_FRAMEBUFFER_COMPLETE, @"Incomplete filter FBO: %d", status);
        #endif
        
        glBindTexture(GL_TEXTURE_2D, 0);
//    });
}

- (void)generateTexture;
{
    glActiveTexture(GL_TEXTURE1);
    glGenTextures(1, &_texture);
    glBindTexture(GL_TEXTURE_2D, _texture);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, _textureOptions.minFilter);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, _textureOptions.magFilter);
    // This is necessary for non-power-of-two textures
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, _textureOptions.wrapS);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, _textureOptions.wrapT);
    
    // TODO: Handle mipmaps
}

- (void)destroyFramebuffer;
{
//    runSynchronouslyOnVideoProcessingQueue(^{
//        [GPUImageContext useImageProcessingContext];
        
        if (framebuffer)
        {
            glDeleteFramebuffers(1, &framebuffer);
            framebuffer = 0;
        }

//
//        if ([GPUImageContext supportsFastTextureUpload] && (!_missingFramebuffer))
//        {
//#if TARGET_IPHONE_SIMULATOR || TARGET_OS_IPHONE
//            if (renderTarget)
//            {
//                CFRelease(renderTarget);
//                renderTarget = NULL;
//            }
//
//            if (renderTexture)
//            {
//                CFRelease(renderTexture);
//                renderTexture = NULL;
//            }
//#endif
//        }
//        else
        {
            glDeleteTextures(1, &_texture);
        }

//    });
}

- (GLuint)texture;
{
//    NSLog(@"Accessing texture: %d from FB: %@", _texture, self);
    return _texture;
}

#pragma mark -
#pragma mark Image capture

void dataProviderReleaseCallback (void *info, const void *data, size_t size)
{
    free((void *)data);
}

- (CGImageRef)newCGImageFromFramebufferContents;
{
    // a CGImage can only be created from a 'normal' color texture
    NSAssert(self.textureOptions.internalFormat == GL_RGBA, @"For conversion to a CGImage the output texture format for this filter must be GL_RGBA.");
    NSAssert(self.textureOptions.type == GL_UNSIGNED_BYTE, @"For conversion to a CGImage the type of the output texture of this filter must be GL_UNSIGNED_BYTE.");
    
    __block CGImageRef cgImageFromBytes;
    
//    runSynchronouslyOnVideoProcessingQueue(^{
//        [GPUImageContext useImageProcessingContext];
        
        NSUInteger totalBytesForImage = (int)_size.width * (int)_size.height * 4;
        // It appears that the width of a texture must be padded out to be a multiple of 8 (32 bytes) if reading from it using a texture cache
        
        GLubyte *rawImagePixels;
        
        CGDataProviderRef dataProvider = NULL;
//        if ([GPUImageContext supportsFastTextureUpload])
//        {
//#if TARGET_IPHONE_SIMULATOR || TARGET_OS_IPHONE
//            NSUInteger paddedWidthOfImage = CVPixelBufferGetBytesPerRow(renderTarget) / 4.0;
//            NSUInteger paddedBytesForImage = paddedWidthOfImage * (int)_size.height * 4;
//
//            glFinish();
//            CFRetain(renderTarget); // I need to retain the pixel buffer here and release in the data source callback to prevent its bytes from being prematurely deallocated during a photo write operation
//            [self lockForReading];
//            rawImagePixels = (GLubyte *)CVPixelBufferGetBaseAddress(renderTarget);
//            dataProvider = CGDataProviderCreateWithData((__bridge_retained void*)self, rawImagePixels, paddedBytesForImage, dataProviderUnlockCallback);
//            [[GPUImageContext sharedFramebufferCache] addFramebufferToActiveImageCaptureList:self]; // In case the framebuffer is swapped out on the filter, need to have a strong reference to it somewhere for it to hang on while the image is in existence
//#else
//#endif
//        }
//        else
        {
            [self activateFramebuffer];
            rawImagePixels = (GLubyte *)malloc(totalBytesForImage);
            glReadPixels(0, 0, (int)_size.width, (int)_size.height, GL_RGBA, GL_UNSIGNED_BYTE, rawImagePixels);
            dataProvider = CGDataProviderCreateWithData(NULL, rawImagePixels, totalBytesForImage, dataProviderReleaseCallback);
//            [self unlock]; // Don't need to keep this around anymore
        }
        
        CGColorSpaceRef defaultRGBColorSpace = CGColorSpaceCreateDeviceRGB();
        
//        if ([GPUImageContext supportsFastTextureUpload])
//        {
//#if TARGET_IPHONE_SIMULATOR || TARGET_OS_IPHONE
//            cgImageFromBytes = CGImageCreate((int)_size.width, (int)_size.height, 8, 32, CVPixelBufferGetBytesPerRow(renderTarget), defaultRGBColorSpace, kCGBitmapByteOrder32Little | kCGImageAlphaPremultipliedFirst, dataProvider, NULL, NO, kCGRenderingIntentDefault);
//#else
//#endif
//        }
//        else
        {
            cgImageFromBytes = CGImageCreate((int)_size.width, (int)_size.height, 8, 32, 4 * (int)_size.width, defaultRGBColorSpace, kCGBitmapByteOrderDefault | kCGImageAlphaLast, dataProvider, NULL, NO, kCGRenderingIntentDefault);
        }
        
        // Capture image with current device orientation
        CGDataProviderRelease(dataProvider);
        CGColorSpaceRelease(defaultRGBColorSpace);
        
//    });
    
    return cgImageFromBytes;
}

- (void)activateFramebuffer;
{
    glBindFramebuffer(GL_FRAMEBUFFER, framebuffer);
    glViewport(0, 0, (int)_size.width, (int)_size.height);
}


@end

//
//  GPUImageOutput.h
//  OpenGLES05-纹理贴图
//
//  Created by libb on 2020/12/28.
//  Copyright © 2020 qinmin. All rights reserved.
//

#import <Foundation/Foundation.h>
#if TARGET_IPHONE_SIMULATOR || TARGET_OS_IPHONE
#import <OpenGLES/EAGL.h>
#import <OpenGLES/ES2/gl.h>
#import <OpenGLES/ES2/glext.h>
#else
#import <OpenGL/OpenGL.h>
#import <OpenGL/gl.h>
#endif

#import <QuartzCore/QuartzCore.h>
#import <CoreMedia/CoreMedia.h>

NS_ASSUME_NONNULL_BEGIN
typedef struct GPUTextureOptions {
    GLenum minFilter;
    GLenum magFilter;
    GLenum wrapS;
    GLenum wrapT;
    GLenum internalFormat;
    GLenum format;
    GLenum type;
} GPUTextureOptions;


@interface GPUImageOutput : NSObject
@property(readwrite, nonatomic) GPUTextureOptions outputTextureOptions;
@end

NS_ASSUME_NONNULL_END

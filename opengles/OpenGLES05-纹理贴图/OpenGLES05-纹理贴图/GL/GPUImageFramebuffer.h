//
//  GPUImageFramebuffer.h
//  OpenGLES05-纹理贴图
//
//  Created by libb on 2020/12/28.
//  Copyright © 2020 qinmin. All rights reserved.
//

#import "GPUImageOutput.h"

NS_ASSUME_NONNULL_BEGIN

@interface GPUImageFramebuffer : NSObject

@property(readonly) CGSize size;
@property(readonly) GPUTextureOptions textureOptions;
@property(readonly) GLuint texture;
@property(readonly) BOOL missingFramebuffer;

- (id)initWithSize:(CGSize)framebufferSize textureOptions:(GPUTextureOptions)fboTextureOptions onlyTexture:(BOOL)onlyGenerateTexture;

// Usage
- (void)activateFramebuffer;
- (CGImageRef)newCGImageFromFramebufferContents;
@end

NS_ASSUME_NONNULL_END

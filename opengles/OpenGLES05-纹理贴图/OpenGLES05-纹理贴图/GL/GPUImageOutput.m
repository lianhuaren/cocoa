//
//  GPUImageOutput.m
//  OpenGLES05-纹理贴图
//
//  Created by libb on 2020/12/28.
//  Copyright © 2020 qinmin. All rights reserved.
//

#import "GPUImageOutput.h"

@implementation GPUImageOutput
- (id)init;
{
    if (!(self = [super init]))
    {
        return nil;
    }

//    targets = [[NSMutableArray alloc] init];
//    targetTextureIndices = [[NSMutableArray alloc] init];
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
@end

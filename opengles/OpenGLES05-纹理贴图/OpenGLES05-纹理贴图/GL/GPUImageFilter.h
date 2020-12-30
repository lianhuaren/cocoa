//
//  GPUImageFilter.h
//  opengles01
//
//  Created by libb on 2020/12/27.
//  Copyright © 2020 Rhythmic Fistman. All rights reserved.
//

#import <Foundation/Foundation.h>
#import "GPUImageContext.h"
NS_ASSUME_NONNULL_BEGIN

#define STRINGIZE(x) #x
#define STRINGIZE2(x) STRINGIZE(x)
#define SHADER_STRING(text) @ STRINGIZE2(text)

#define GPUImageHashIdentifier #
#define GPUImageWrappedLabel(x) x
#define GPUImageEscapedHashIdentifier(a) GPUImageWrappedLabel(GPUImageHashIdentifier)a

extern NSString *const kGPUImageVertexShaderString;
extern NSString *const kGPUImagePassthroughFragmentShaderString;


@interface GPUImageFilter : NSObject
+ (const GLfloat *)textureCoordinatesForRotation:(GPUImageRotationMode)rotationMode;
@end

NS_ASSUME_NONNULL_END

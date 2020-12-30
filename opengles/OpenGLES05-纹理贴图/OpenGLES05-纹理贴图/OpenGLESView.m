//
//  OpenGLESView.m
//  OpenGLES01-环境搭建
//
//  Created by qinmin on 2017/2/9.
//  Copyright © 2017年 qinmin. All rights reserved.
//

#import "OpenGLESView.h"
#import <OpenGLES/ES2/gl.h>
#import "GLUtil.h"
#include "JpegUtil.h"
#import "GPUImageContext.h"
#import "GPUImageFilter.h"
#import "GPUImageFramebuffer.h"

extern void dataProviderReleaseCallback (void *info, const void *data, size_t size);
@interface OpenGLESView ()
{
    CAEAGLLayer     *_eaglLayer;
    EAGLContext     *_context;
//    GLuint          _colorRenderBuffer;
//    GLuint          _frameBuffer;
//
//    GLuint          _program;
    GLuint          _vbo;
    GLuint          _image_texture;
    int             _vertCount;
    
    GLuint displayRenderbuffer, displayFramebuffer;
    GLProgram *displayProgram;
    
    GLint displayPositionAttribute, displayTextureCoordinateAttribute;
    GLint displayInputTextureUniform;
    
    GPUImageFramebuffer *outputFramebuffer;
    
    
    GLuint          _outframeBufferForNext;
    GLuint          _program1;
    GLuint          _vbo1;
    GLuint          outTexture;
}

@property(readwrite, nonatomic) GPUTextureOptions outputTextureOptions;

@end

@implementation OpenGLESView

+ (Class)layerClass
{
    // 只有 [CAEAGLLayer class] 类型的 layer 才支持在其上描绘 OpenGL 内容。
    return [CAEAGLLayer class];
}

- (void)dealloc
{
    glDeleteBuffers(1, &_vbo);
    glDeleteTextures(1, &_image_texture);
//    glDeleteProgram(_program);
}

- (instancetype)initWithFrame:(CGRect)frame
{
    if (self = [super initWithFrame:frame]) {
        [self setupLayer];
        [self setupContext];
        [self setupGLProgram];
        [self setupGLProgram1];
//        [self setupVBO];
//        [self setupTexure];
    }
    return self;
}

- (void)layoutSubviews
{
    [EAGLContext setCurrentContext:_context];
    
    [self destoryRenderAndFrameBuffer];
    
    [self setupFrameAndRenderBuffer];
    
    [self setupFrameBuffer1];
    
    [self render];
}


#pragma mark - Setup
- (void)setupLayer
{
    _eaglLayer = (CAEAGLLayer*) self.layer;
    
    // CALayer 默认是透明的，必须将它设为不透明才能让其可见
    _eaglLayer.opaque = YES;
    
    // 设置描绘属性，在这里设置不维持渲染内容以及颜色格式为 RGBA8
    _eaglLayer.drawableProperties = [NSDictionary dictionaryWithObjectsAndKeys:
                                     [NSNumber numberWithBool:NO], kEAGLDrawablePropertyRetainedBacking, kEAGLColorFormatRGBA8, kEAGLDrawablePropertyColorFormat, nil];
}

- (void)setupContext
{
    // 设置OpenGLES的版本为2.0 当然还可以选择1.0和最新的3.0的版本，以后我们会讲到2.0与3.0的差异，目前为了兼容性选择2.0的版本
    _context = [[EAGLContext alloc] initWithAPI:kEAGLRenderingAPIOpenGLES2];
    if (!_context) {
        NSLog(@"Failed to initialize OpenGLES 2.0 context");
        exit(1);
    }
    
    // 将当前上下文设置为我们创建的上下文
    if (![EAGLContext setCurrentContext:_context]) {
        NSLog(@"Failed to set current OpenGL context");
        exit(1);
    }
}

- (void)setupFrameAndRenderBuffer
{
//    glGenRenderbuffers(1, &_colorRenderBuffer);
//    glBindRenderbuffer(GL_RENDERBUFFER, _colorRenderBuffer);
//    // 为 color renderbuffer 分配存储空间
//    [_context renderbufferStorage:GL_RENDERBUFFER fromDrawable:_eaglLayer];
//
//    glGenFramebuffers(1, &_frameBuffer);
//    // 设置为当前 framebuffer
//    glBindFramebuffer(GL_FRAMEBUFFER, _frameBuffer);
//    // 将 _colorRenderBuffer 装配到 GL_COLOR_ATTACHMENT0 这个装配点上
//    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
//                              GL_RENDERBUFFER, _colorRenderBuffer);
    
    glGenRenderbuffers(1, &displayRenderbuffer);
    glBindRenderbuffer(GL_RENDERBUFFER, displayRenderbuffer);

//    [[[GPUImageContext sharedImageProcessingContext] context] renderbufferStorage:GL_RENDERBUFFER fromDrawable:(CAEAGLLayer*)self.layer];
    [_context renderbufferStorage:GL_RENDERBUFFER fromDrawable:_eaglLayer];
    
    glGenFramebuffers(1, &displayFramebuffer);
    glBindFramebuffer(GL_FRAMEBUFFER, displayFramebuffer);

    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_RENDERBUFFER, displayRenderbuffer);

}

- (void)setupFrameBuffer1
{
//    _texture1 = createTexture2D(GL_RGBA, self.frame.size.width, self.frame.size.height, NULL);
//    glGenFramebuffers(1, &_frameBuffer1);
//    // 设置为当前 framebuffer
//    glBindFramebuffer(GL_FRAMEBUFFER, _frameBuffer1);
//    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, _texture1, 0);
}


- (void)setupGLProgram
{
    NSString *vertFile = [[NSBundle mainBundle] pathForResource:@"vert.glsl" ofType:nil];
    NSString *fragFile = [[NSBundle mainBundle] pathForResource:@"frag.glsl" ofType:nil];
//    _program = createGLProgramFromFile(vertFile.UTF8String, fragFile.UTF8String);
//
//    glUseProgram(_program);
    
    NSError *error;
    NSString *vertFileStr = [[NSString alloc] initWithContentsOfFile:vertFile encoding:NSUTF8StringEncoding error:&error];
    if (vertFileStr == nil) {
        NSLog(@"Failed to load vertex shader: %@", [error localizedDescription]);
        return;
    }
    NSString *fragFileStr = [[NSString alloc] initWithContentsOfFile:fragFile encoding:NSUTF8StringEncoding error:&error];
    if (fragFileStr == nil) {
        NSLog(@"Failed to load vertex shader: %@", [error localizedDescription]);
        return;
    }

    displayProgram = [[GPUImageContext sharedImageProcessingContext] programForVertexShaderString:vertFileStr fragmentShaderString:fragFileStr];
//    displayProgram = [[GPUImageContext sharedImageProcessingContext] programForVertexShaderString:kGPUImageVertexShaderString fragmentShaderString:kGPUImagePassthroughFragmentShaderString];
    if (!displayProgram.initialized)
    {
        [displayProgram addAttribute:@"position"];
        [displayProgram addAttribute:@"inputTextureCoordinate"];

        
        if (![displayProgram link])
        {
            NSString *progLog = [displayProgram programLog];
            NSLog(@"Program link log: %@", progLog);
            NSString *fragLog = [displayProgram fragmentShaderLog];
            NSLog(@"Fragment shader compile log: %@", fragLog);
            NSString *vertLog = [displayProgram vertexShaderLog];
            NSLog(@"Vertex shader compile log: %@", vertLog);
            displayProgram = nil;
            NSAssert(NO, @"Filter shader link failed");
        }
    }

}

- (void)setupGLProgram1
{
    NSString *vertFile = [[NSBundle mainBundle] pathForResource:@"vert.glsl" ofType:nil];
    NSString *fragFile = [[NSBundle mainBundle] pathForResource:@"frag.glsl" ofType:nil];
    _program1 = createGLProgramFromFile(vertFile.UTF8String, fragFile.UTF8String);
}

- (void)setupVBO
{
    _vertCount = 6;
    
    GLfloat vertices[] = {
        0.5f,  0.5f, 0.0f, 1.0f, 0.0f,   // 右上
        0.5f, -0.5f, 0.0f, 1.0f, 1.0f,   // 右下
        -0.5f, -0.5f, 0.0f, 0.0f, 1.0f,  // 左下
        -0.5f,  0.5f, 0.0f, 0.0f, 0.0f   // 左上
    };
    
//    GLfloat vertices[] = {
//        0.5f,  0.5f, 0.0f, 1.0f, 0.0f,   // 右上
//        0.5f, -0.5f, 0.0f, 1.0f, 1.0f,   // 右下
//        -0.5f, -0.5f, 0.0f, 0.0f, 1.0f,  // 左下
//        -0.5f, -0.5f, 0.0f, 0.0f, 1.0f,  // 左下
//        -0.5f,  0.5f, 0.0f, 0.0f, 0.0f,  // 左上
//        0.5f,  0.5f, 0.0f, 1.0f, 0.0f,   // 右上
//    };
    
    // 创建VBO
    _vbo = createVBO(GL_ARRAY_BUFFER, GL_STATIC_DRAW, sizeof(vertices), vertices);
    
//    glEnableVertexAttribArray(glGetAttribLocation(_program, "position"));
//    glVertexAttribPointer(glGetAttribLocation(_program, "position"), 3, GL_FLOAT, GL_FALSE, sizeof(GLfloat)*5, NULL);
    
    
//    glEnableVertexAttribArray(glGetAttribLocation(_program, "texcoord"));
//    glVertexAttribPointer(glGetAttribLocation(_program, "texcoord"), 2, GL_FLOAT, GL_FALSE, sizeof(GLfloat)*5, NULL+sizeof(GL_FLOAT)*3);
    
    displayPositionAttribute = [displayProgram attributeIndex:@"position"];
    displayTextureCoordinateAttribute = [displayProgram attributeIndex:@"inputTextureCoordinate"];

    glEnableVertexAttribArray(displayPositionAttribute);
    glEnableVertexAttribArray(displayTextureCoordinateAttribute);
    
    glVertexAttribPointer(displayPositionAttribute, 3, GL_FLOAT, 0, sizeof(GLfloat)*5, NULL);
    glVertexAttribPointer(displayTextureCoordinateAttribute, 2, GL_FLOAT, 0, sizeof(GLfloat)*5, NULL+sizeof(GL_FLOAT)*3);

}

- (void)setupVBO1
{
    _vertCount = 6;
    
    GLfloat vertices[] = {
        1.0f,  1.0f, 0.0f, 1.0f, 1.0f,   // 右上
        1.0f, -1.0f, 0.0f, 1.0f, 0.0f,   // 右下
        -1.0f, -1.0f, 0.0f, 0.0f, 0.0f,  // 左下
        -1.0f,  1.0f, 0.0f, 0.0f, 1.0f   // 左上
    };
    
    // 创建VBO
    _vbo1 = createVBO(GL_ARRAY_BUFFER, GL_STATIC_DRAW, sizeof(vertices), vertices);
    
    glEnableVertexAttribArray(glGetAttribLocation(_program1, "position"));
    glVertexAttribPointer(glGetAttribLocation(_program1, "position"), 3, GL_FLOAT, GL_FALSE, sizeof(GLfloat)*5, NULL);
    
    glEnableVertexAttribArray(glGetAttribLocation(_program1, "texcoord"));
    glVertexAttribPointer(glGetAttribLocation(_program1, "texcoord"), 2, GL_FLOAT, GL_FALSE, sizeof(GLfloat)*5, NULL+sizeof(GL_FLOAT)*3);
}



- (void)setupTexure
{
    NSString *path = [[NSBundle mainBundle] pathForResource:@"wood" ofType:@"jpg"];
    
    unsigned char *data;
    int size;
    int width;
    int height;
    
    // 加载纹理
    if (read_jpeg_file(path.UTF8String, &data, &size, &width, &height) < 0) {
        printf("%s\n", "decode fail");
    }
//    if (data) {
//        free(data);
//        data = NULL;
//    }
    
//    data = malloc(width*height*4);
//    memset(data, 177, width*height*4);
    // 创建纹理
//    _texture = createTexture2D(GL_RGB, width, height, data);
    {
        GLenum internalFormat = GL_RGB;
        GLenum format = GL_RGB;
        glGenTextures(1, &_image_texture);
        glBindTexture(GL_TEXTURE_2D, _image_texture);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glTexImage2D(GL_TEXTURE_2D, 0, internalFormat, width, height, 0, format, GL_UNSIGNED_BYTE, data);
        glBindTexture(GL_TEXTURE_2D, 0);
        
        

        
        if (0) {
            
            outputFramebuffer = [[GPUImageFramebuffer alloc] initWithSize:CGSizeMake(width, height) textureOptions:self.outputTextureOptions onlyTexture:NO];
            [outputFramebuffer activateFramebuffer];

            
            glTexImage2D(GL_TEXTURE_2D, 0, internalFormat, width, height, 0, format, GL_UNSIGNED_BYTE, data);
            glBindTexture(GL_TEXTURE_2D, 0);
            
//            glActiveTexture(GL_TEXTURE0);
//            glBindTexture(GL_TEXTURE_2D, _texture);
//            glUniform1i(displayInputTextureUniform, 0);
//            glUniform1i(displayInputTextureUniform, 1);
//
//            glDrawArrays(GL_TRIANGLES, 0, _vertCount);
            
            

        }
        
//        if (0) {
////        outputFramebuffer = [self fetchFramebufferForSize:CGSizeMake(width, height) textureOptions:self.outputTextureOptions onlyTexture:NO];
        
//        
////        glActiveTexture(GL_TEXTURE1);
//            glGenTextures(1, &_texture);
//            glBindTexture(GL_TEXTURE_2D, _texture);
//            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, _outputTextureOptions.minFilter);
//            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, _outputTextureOptions.magFilter);
//            // This is necessary for non-power-of-two textures
//            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, _outputTextureOptions.wrapS);
//            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, _outputTextureOptions.wrapT);
//            
//            glTexImage2D(GL_TEXTURE_2D, 0, _outputTextureOptions.internalFormat, (int)width, (int)height, 0, _outputTextureOptions.format, _outputTextureOptions.type, data);
//            glBindTexture(GL_TEXTURE_2D, 0);
//        
//        }
    }
    
    if (data) {
        free(data);
        data = NULL;
    }
}

#pragma mark - Clean
- (void)destoryRenderAndFrameBuffer
{
//    glDeleteFramebuffers(1, &_frameBuffer);
//    _frameBuffer = 0;
//    glDeleteRenderbuffers(1, &_colorRenderBuffer);
//    _colorRenderBuffer = 0;
    if (displayFramebuffer)
    {
        glDeleteFramebuffers(1, &displayFramebuffer);
        displayFramebuffer = 0;
    }
    
    if (displayRenderbuffer)
    {
        glDeleteRenderbuffers(1, &displayRenderbuffer);
        displayRenderbuffer = 0;
    }
}

#pragma mark - Render
- (void)render
{
    /************************离屏渲染********************************************/
//    {
//        glGenFramebuffers(1, &_outframeBufferForNext);
//        // 设置为当前 framebuffer
//        glBindFramebuffer(GL_FRAMEBUFFER, _outframeBufferForNext);
//    }
    _outputTextureOptions.minFilter = GL_LINEAR;
    _outputTextureOptions.magFilter = GL_LINEAR;
    _outputTextureOptions.wrapS = GL_CLAMP_TO_EDGE;
    _outputTextureOptions.wrapT = GL_CLAMP_TO_EDGE;
    _outputTextureOptions.internalFormat = GL_RGBA;
    _outputTextureOptions.format = GL_RGBA;
    _outputTextureOptions.type = GL_UNSIGNED_BYTE;
    
    outputFramebuffer = [[GPUImageFramebuffer alloc] initWithSize:CGSizeMake(self.frame.size.width, self.frame.size.height) textureOptions:self.outputTextureOptions onlyTexture:NO];
    [outputFramebuffer activateFramebuffer];

    
    glClearColor(1.0, 1.0, 1.0, 1.0);
    glClear(GL_COLOR_BUFFER_BIT);
    glLineWidth(2.0);
    
    glViewport(0, 0, self.frame.size.width, self.frame.size.height);
    
    
    [self setupTexure];

    
//    outTexture = createTexture2D(GL_RGBA, self.frame.size.width, self.frame.size.height, NULL);
//    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, outTexture, 0);
    
    
    glUseProgram(_program1);//setActiveShaderProgram
    
    // 激活纹理
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, _image_texture);
    glUniform1i(glGetUniformLocation(_program1, "image"), 0);
    
    [self setupVBO1];
    // 索引数组
    unsigned int indices1[] = {0,1,2,3,2,0};
    glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, indices1);
    
    CGImageRef imageRef = [outputFramebuffer newCGImageFromFramebufferContents];
    UIImage *image = [UIImage imageWithCGImage:imageRef];
    NSLog(@"image:%@",image);
    
    
    
    glBindFramebuffer(GL_FRAMEBUFFER, displayFramebuffer);//[GPUImageFramebuffer alloc]
    
    glClearColor(1.0, 1.0, 1.0, 1.0);
    glClear(GL_COLOR_BUFFER_BIT);
    glLineWidth(2.0);
    
    glViewport(0, 0, self.frame.size.width, self.frame.size.height);
    
    
//    _texture1 = createTexture2D(GL_RGBA, self.frame.size.width, self.frame.size.height, NULL);
//    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, _texture1, 0);
    
    [GPUImageContext setActiveShaderProgram:displayProgram];
    // 激活纹理
    glActiveTexture(GL_TEXTURE4);
    glBindTexture(GL_TEXTURE_2D, [outputFramebuffer texture]);
    
//    glUniform1i(glGetUniformLocation(_program, "image"), 1);
    displayInputTextureUniform = [displayProgram uniformIndex:@"inputImageTexture"];
    glUniform1i(displayInputTextureUniform, 4);
    
//    [self setupVBO];
    {
        static GLfloat vertices[] = {
//            0.5f,  0.5f,// 右上
//            0.5f, -0.5f,  // 右下
//            -0.5f, -0.5f,   // 左下
//            -0.5f,  0.5f,    // 左上
//            0.5f,  0.5f   // 右上
            -1.0f, -1.0f,//左下
            1.0f, -1.0f,// 右下
            -1.0f,  1.0f,// 左上
            1.0f,  1.0f,// 右上
        };
//        GLfloat vertices[] = {
//            0.5f,  0.5f, 0.0f, 1.0f, 0.0f,   // 右上
//            0.5f, -0.5f, 0.0f, 1.0f, 1.0f,   // 右下
//            -0.5f, -0.5f, 0.0f, 0.0f, 1.0f,  // 左下
//            -0.5f,  0.5f, 0.0f, 0.0f, 0.0f,   // 左上
//            0.5f,  0.5f, 0.0f, 1.0f, 0.0f   // 右上
//        };
        
        // 创建VBO
        _vbo = createVBO(GL_ARRAY_BUFFER, GL_STATIC_DRAW, sizeof(vertices), vertices);
//        GLuint vbo;
//        glGenBuffers(1, &vbo);
//        glBindBuffer(GL_ARRAY_BUFFER, vbo);
//        glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
        
        displayPositionAttribute = [displayProgram attributeIndex:@"position"];
        displayTextureCoordinateAttribute = [displayProgram attributeIndex:@"inputTextureCoordinate"];

        glEnableVertexAttribArray(displayPositionAttribute);
//        glEnableVertexAttribArray(displayTextureCoordinateAttribute);
        
        glVertexAttribPointer(displayPositionAttribute, 2, GL_FLOAT, 0, sizeof(GLfloat)*2, NULL);
//        glVertexAttribPointer(displayTextureCoordinateAttribute, 2, GL_FLOAT, 0, sizeof(GLfloat)*5, NULL+sizeof(GL_FLOAT)*3);
        
        
        static GLfloat verticescord[] = {
            0.0f, 0.0f,
            1.0f, 0.0f,
            0.0f, 1.0f,
            1.0f, 1.0f,
//            0.0f, 1.0f,  // 左下
//            1.0f, 1.0f,   // 右下
//            0.0f, 0.0f,   // 左上
//            1.0f, 0.0f   // 右上

        };
        
        createVBO(GL_ARRAY_BUFFER, GL_STATIC_DRAW, sizeof(GLfloat)*8, verticescord);
        glEnableVertexAttribArray(displayTextureCoordinateAttribute);
        glVertexAttribPointer(displayTextureCoordinateAttribute, 2, GL_FLOAT, 0, sizeof(GLfloat)*2, NULL);
         //索引数组
//        unsigned int indices[] = {0,1,2,3,2,0};
//        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, indices);
        
//        static const GLfloat imageVertices[] = {
//            -1.0f, -1.0f,
//            1.0f, -1.0f,
//            -1.0f,  1.0f,
//            1.0f,  1.0f,
//        };
//        glVertexAttribPointer(displayPositionAttribute, 2, GL_FLOAT, 0, 0, imageVertices);
        
        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
    }
    
//    CGSize _size = CGSizeMake(self.frame.size.width, self.frame.size.height);
//    glBindFramebuffer(GL_FRAMEBUFFER, outFramebufferForNext);
//
//
//    glViewport(0, 0, (int)_size.width, (int)_size.height);
//
//    __block CGImageRef cgImageFromBytes;
//
//    NSUInteger totalBytesForImage = (int)_size.width * (int)_size.height * 4;
//    // It appears that the width of a texture must be padded out to be a multiple of 8 (32 bytes) if reading from it using a texture cache
//
//    GLubyte *rawImagePixels;
//
//    CGDataProviderRef dataProvider = NULL;
//
//    rawImagePixels = (GLubyte *)malloc(totalBytesForImage);
//    glReadPixels(0, 0, (int)_size.width, (int)_size.height, GL_RGBA, GL_UNSIGNED_BYTE, rawImagePixels);
//    dataProvider = CGDataProviderCreateWithData(NULL, rawImagePixels, totalBytesForImage, dataProviderReleaseCallback);
//
//
//    CGColorSpaceRef defaultRGBColorSpace = CGColorSpaceCreateDeviceRGB();
//    {
//        cgImageFromBytes = CGImageCreate((int)_size.width, (int)_size.height, 8, 32, 4 * (int)_size.width, defaultRGBColorSpace, kCGBitmapByteOrderDefault | kCGImageAlphaLast, dataProvider, NULL, NO, kCGRenderingIntentDefault);
//    }
//
//    // Capture image with current device orientation
//    CGDataProviderRelease(dataProvider);
//    CGColorSpaceRelease(defaultRGBColorSpace);
//
////    CGImageRef imageRef = [outputFramebuffer newCGImageFromFramebufferContents];
//    UIImage *image = [UIImage imageWithCGImage:cgImageFromBytes];
//    NSLog(@"image:%@",image);
    
    if (0){
        
        CGFloat heightScaling, widthScaling;
        GLfloat imageVertices[8];
        
        widthScaling = 1.0;
        heightScaling = 1.0;
        
        imageVertices[0] = -widthScaling;
        imageVertices[1] = -heightScaling;
        imageVertices[2] = widthScaling;
        imageVertices[3] = -heightScaling;
        imageVertices[4] = -widthScaling;
        imageVertices[5] = heightScaling;
        imageVertices[6] = widthScaling;
        imageVertices[7] = heightScaling;
        
        glVertexAttribPointer(displayPositionAttribute, 2, GL_FLOAT, 0, 0, imageVertices);
//        glVertexAttribPointer(displayTextureCoordinateAttribute, 2, GL_FLOAT, 0, 0, [GPUImageView textureCoordinatesForRotation:inputRotation]);
        
        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
    }
    

    
    //将指定 renderbuffer 呈现在屏幕上，在这里我们指定的是前面已经绑定为当前 renderbuffer 的那个，在 renderbuffer 可以被呈现之前，必须调用renderbufferStorage:fromDrawable: 为之分配存储空间。
    [_context presentRenderbuffer:GL_RENDERBUFFER];
    
//    CGImageRef imageRef = [self newCGImageFromFramebufferContents];
//    UIImage *image = [UIImage imageWithCGImage:imageRef];
//    NSLog(@"image:%@",image);
    
    return;
//    if (0) {
//        glClearColor(1.0, 1.0, 0, 1.0);
//        glClear(GL_COLOR_BUFFER_BIT);
//        glLineWidth(2.0);
//
//        glViewport(0, 0, self.frame.size.width, self.frame.size.height);
//
//        [self setupVBO];
//        [self setupTexure];
//
//        // 激活纹理
//    //    glActiveTexture(GL_TEXTURE0);
//    //    glBindTexture(GL_TEXTURE_2D, _texture);
//    //    glUniform1i(glGetUniformLocation(_program, "image"), 0);
//
//        glActiveTexture(GL_TEXTURE0);
//        glBindTexture(GL_TEXTURE_2D, _image_texture);
//    //    glBindTexture(GL_TEXTURE_2D, [outputFramebuffer texture]);
//        glUniform1i(displayInputTextureUniform, 0);
//
//
//        glDrawArrays(GL_TRIANGLES, 0, _vertCount);
//
//
//
//        // 索引数组
//        unsigned int indices[] = {0,1,2,3,2,0};
//        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, indices);
//
//        //将指定 renderbuffer 呈现在屏幕上，在这里我们指定的是前面已经绑定为当前 renderbuffer 的那个，在 renderbuffer 可以被呈现之前，必须调用renderbufferStorage:fromDrawable: 为之分配存储空间。
//        [_context presentRenderbuffer:GL_RENDERBUFFER];
//    }
    
}




- (GPUImageFramebuffer *)fetchFramebufferForSize:(CGSize)framebufferSize textureOptions:(GPUTextureOptions)textureOptions onlyTexture:(BOOL)onlyTexture;
{
    __block GPUImageFramebuffer *framebufferFromCache = nil;
//    dispatch_sync(framebufferCacheQueue, ^{
//    runSynchronouslyOnVideoProcessingQueue(^{
//        NSString *lookupHash = [self hashForSize:framebufferSize textureOptions:textureOptions onlyTexture:onlyTexture];
//        NSNumber *numberOfMatchingTexturesInCache = [framebufferTypeCounts objectForKey:lookupHash];
//        NSInteger numberOfMatchingTextures = [numberOfMatchingTexturesInCache integerValue];
//
//        if ([numberOfMatchingTexturesInCache integerValue] < 1)
//        {
//            // Nothing in the cache, create a new framebuffer to use
            framebufferFromCache = [[GPUImageFramebuffer alloc] initWithSize:framebufferSize textureOptions:textureOptions onlyTexture:onlyTexture];
//        }
//        else
//        {
//            // Something found, pull the old framebuffer and decrement the count
//            NSInteger currentTextureID = (numberOfMatchingTextures - 1);
//            while ((framebufferFromCache == nil) && (currentTextureID >= 0))
//            {
//                NSString *textureHash = [NSString stringWithFormat:@"%@-%ld", lookupHash, (long)currentTextureID];
//                framebufferFromCache = [framebufferCache objectForKey:textureHash];
//                // Test the values in the cache first, to see if they got invalidated behind our back
//                if (framebufferFromCache != nil)
//                {
//                    // Withdraw this from the cache while it's in use
//                    [framebufferCache removeObjectForKey:textureHash];
//                }
//                currentTextureID--;
//            }
//
//            currentTextureID++;
//
//            [framebufferTypeCounts setObject:[NSNumber numberWithInteger:currentTextureID] forKey:lookupHash];
//
//            if (framebufferFromCache == nil)
//            {
//                framebufferFromCache = [[GPUImageFramebuffer alloc] initWithSize:framebufferSize textureOptions:textureOptions onlyTexture:onlyTexture];
//            }
//        }
//    });

//    [framebufferFromCache lock];
    return framebufferFromCache;
}

@end

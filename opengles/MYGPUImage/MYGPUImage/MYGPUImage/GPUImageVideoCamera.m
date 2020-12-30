//
//  GPUImageVideoCamera.m
//  opengles01
//
//  Created by libb on 2020/12/27.
//  Copyright © 2020 Rhythmic Fistman. All rights reserved.
//

#import "GPUImageVideoCamera.h"

#import "GPUImageOutput.h"
#import "GLProgram.h"
#import "GPUImageFilter.h"
#import "GPUImageColorConversion.h"
#import "GPUImageFramebuffer.h"
@interface GPUImageVideoCamera()<AVCaptureVideoDataOutputSampleBufferDelegate>
{
    
    GLProgram *yuvConversionProgram;
    GLint yuvConversionPositionAttribute, yuvConversionTextureCoordinateAttribute;
    GLint yuvConversionLuminanceTextureUniform, yuvConversionChrominanceTextureUniform;
    GLint yuvConversionMatrixUniform;
    const GLfloat *_preferredConversion;
    
    
}

@end

@implementation GPUImageVideoCamera
@synthesize outputImageOrientation = _outputImageOrientation;

- (id)initWithSessionPreset:(NSString *)sessionPreset cameraPosition:(AVCaptureDevicePosition)cameraPosition;
{
    if (!(self = [super init]))
    {
        return nil;
    }
    captureAsYUV = YES;
    
    
    // Grab the back-facing or front-facing camera
    _inputCamera = nil;
    NSArray *devices = [AVCaptureDevice devicesWithMediaType:AVMediaTypeVideo];
    for (AVCaptureDevice *device in devices)
    {
        if ([device position] == cameraPosition)
        {
            _inputCamera = device;
        }
    }
    
    if (!_inputCamera) {
        return nil;
    }
    // Create the capture session
    _captureSession = [[AVCaptureSession alloc] init];
    
    [_captureSession beginConfiguration];
    
    // Add the video input
    NSError *error = nil;
    videoInput = [[AVCaptureDeviceInput alloc] initWithDevice:_inputCamera error:&error];
    if ([_captureSession canAddInput:videoInput])
    {
        [_captureSession addInput:videoInput];
    }
    
    // Add the video frame output
    videoOutput = [[AVCaptureVideoDataOutput alloc] init];
    [videoOutput setAlwaysDiscardsLateVideoFrames:NO];
    
//    AVCaptureConnection* connection = [videoOutput connectionWithMediaType:AVMediaTypeVideo];
//    connection.videoOrientation = AVCaptureVideoOrientationPortrait;
//    // 调节摄像头翻转
//    connection.videoMirrored = (cameraPosition == AVCaptureDevicePositionFront);
    
    cameraProcessingQueue = dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_HIGH,0);
    audioProcessingQueue = dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_LOW,0);

    [videoOutput setSampleBufferDelegate:self queue:cameraProcessingQueue];
    
    if (captureAsYUV && [GPUImageContext supportsFastTextureUpload])
    {
        BOOL supportsFullYUVRange = NO;
//        NSArray *supportedPixelFormats = videoOutput.availableVideoCVPixelFormatTypes;
//        for (NSNumber *currentPixelFormat in supportedPixelFormats)
//        {
//            if ([currentPixelFormat intValue] == kCVPixelFormatType_420YpCbCr8BiPlanarFullRange)
//            {
//                supportsFullYUVRange = YES;
//            }
//        }
        
        if (supportsFullYUVRange)
        {
            [videoOutput setVideoSettings:[NSDictionary dictionaryWithObject:[NSNumber numberWithInt:kCVPixelFormatType_420YpCbCr8BiPlanarFullRange] forKey:(id)kCVPixelBufferPixelFormatTypeKey]];
            isFullYUVRange = YES;
        }
        else
        {
            [videoOutput setVideoSettings:[NSDictionary dictionaryWithObject:[NSNumber numberWithInt:kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange] forKey:(id)kCVPixelBufferPixelFormatTypeKey]];
            isFullYUVRange = NO;
        }
    }
    
    if ([_captureSession canAddOutput:videoOutput])
    {
        [_captureSession addOutput:videoOutput];
    }
    else
    {
        NSLog(@"Couldn't add video output");
        return nil;
    }
    
    _captureSessionPreset = sessionPreset;
    [_captureSession setSessionPreset:_captureSessionPreset];
    
    [_captureSession commitConfiguration];

    runSynchronouslyOnVideoProcessingQueue(^{
        
        if (captureAsYUV)
        {
            [GPUImageContext useImageProcessingContext];
            //            if ([GPUImageContext deviceSupportsRedTextures])
            //            {
            //                yuvConversionProgram = [[GPUImageContext sharedImageProcessingContext] programForVertexShaderString:kGPUImageVertexShaderString fragmentShaderString:kGPUImageYUVVideoRangeConversionForRGFragmentShaderString];
            //            }
            //            else
            //            {
            if (isFullYUVRange)
            {
                yuvConversionProgram = [[GPUImageContext sharedImageProcessingContext] programForVertexShaderString:kGPUImageVertexShaderString fragmentShaderString:kGPUImageYUVFullRangeConversionForLAFragmentShaderString];
            }
            else
            {
                yuvConversionProgram = [[GPUImageContext sharedImageProcessingContext] programForVertexShaderString:kGPUImageVertexShaderString fragmentShaderString:kGPUImageYUVVideoRangeConversionForLAFragmentShaderString];
            }

            //            }
            
            if (!yuvConversionProgram.initialized)
            {
                [yuvConversionProgram addAttribute:@"position"];
                [yuvConversionProgram addAttribute:@"inputTextureCoordinate"];
                
                if (![yuvConversionProgram link])
                {
                    NSString *progLog = [yuvConversionProgram programLog];
                    NSLog(@"Program link log: %@", progLog);
                    NSString *fragLog = [yuvConversionProgram fragmentShaderLog];
                    NSLog(@"Fragment shader compile log: %@", fragLog);
                    NSString *vertLog = [yuvConversionProgram vertexShaderLog];
                    NSLog(@"Vertex shader compile log: %@", vertLog);
                    yuvConversionProgram = nil;
                    NSAssert(NO, @"Filter shader link failed");
                }
            }
            
            yuvConversionPositionAttribute = [yuvConversionProgram attributeIndex:@"position"];
            yuvConversionTextureCoordinateAttribute = [yuvConversionProgram attributeIndex:@"inputTextureCoordinate"];
            yuvConversionLuminanceTextureUniform = [yuvConversionProgram uniformIndex:@"luminanceTexture"];
            yuvConversionChrominanceTextureUniform = [yuvConversionProgram uniformIndex:@"chrominanceTexture"];
            yuvConversionMatrixUniform = [yuvConversionProgram uniformIndex:@"colorConversionMatrix"];
            
            [GPUImageContext setActiveShaderProgram:yuvConversionProgram];
            
            glEnableVertexAttribArray(yuvConversionPositionAttribute);
            glEnableVertexAttribArray(yuvConversionTextureCoordinateAttribute);
        }
    });
    return self;
}

#pragma mark -
#pragma mark AVCaptureVideoDataOutputSampleBufferDelegate

- (void)captureOutput:(AVCaptureOutput *)captureOutput didOutputSampleBuffer:(CMSampleBufferRef)sampleBuffer fromConnection:(AVCaptureConnection *)connection
{
    if (!self.captureSession.isRunning)
    {
        return;
    }
//    else if (captureOutput == audioOutput)
//    {
//        [self processAudioSampleBuffer:sampleBuffer];
//    }
    else
    {
//        if (dispatch_semaphore_wait(frameRenderingSemaphore, DISPATCH_TIME_NOW) != 0)
//        {
//            return;
//        }
        
        CFRetain(sampleBuffer);
        runAsynchronouslyOnVideoProcessingQueue(^{
            //Feature Detection Hook.
//            if (self.delegate)
//            {
//                [self.delegate willOutputSampleBuffer:sampleBuffer];
//            }
            
            [self processVideoSampleBuffer:sampleBuffer];
            
            CFRelease(sampleBuffer);
//            dispatch_semaphore_signal(frameRenderingSemaphore);
        });
    }
}

- (void)processVideoSampleBuffer:(CMSampleBufferRef)sampleBuffer
{
    CFAbsoluteTime startTime = CFAbsoluteTimeGetCurrent();
    CVImageBufferRef cameraFrame = CMSampleBufferGetImageBuffer(sampleBuffer);
    int bufferWidth = (int) CVPixelBufferGetWidth(cameraFrame);
    int bufferHeight = (int) CVPixelBufferGetHeight(cameraFrame);
    CFTypeRef colorAttachments = CVBufferGetAttachment(cameraFrame, kCVImageBufferYCbCrMatrixKey, NULL);
    if (colorAttachments != NULL)
    {
        if(CFStringCompare(colorAttachments, kCVImageBufferYCbCrMatrix_ITU_R_601_4, 0) == kCFCompareEqualTo)
        {
            if (isFullYUVRange)
            {
                _preferredConversion = kColorConversion601FullRange;
            }
            else
            {
                _preferredConversion = kColorConversion601;
            }
        }
        else
        {
            _preferredConversion = kColorConversion709;
        }
    }
    else
    {
        if (isFullYUVRange)
        {
            _preferredConversion = kColorConversion601FullRange;
        }
        else
        {
            _preferredConversion = kColorConversion601;
        }
    }
    
    CMTime currentTime = CMSampleBufferGetPresentationTimeStamp(sampleBuffer);

    [GPUImageContext useImageProcessingContext];

    if ([GPUImageContext supportsFastTextureUpload] && captureAsYUV)
    {
        CVOpenGLESTextureRef luminanceTextureRef = NULL;
        CVOpenGLESTextureRef chrominanceTextureRef = NULL;
        
        if (CVPixelBufferGetPlaneCount(cameraFrame) > 0) // Check for YUV planar inputs to do RGB conversion
        {
            CVPixelBufferLockBaseAddress(cameraFrame, 0);
            
            if ( (imageBufferWidth != bufferWidth) && (imageBufferHeight != bufferHeight) )
            {
                imageBufferWidth = bufferWidth;
                imageBufferHeight = bufferHeight;
            }
            
            CVReturn err;
            // Y-plane
            glActiveTexture(GL_TEXTURE4);
            if ([GPUImageContext deviceSupportsRedTextures])
            {
//                err = CVOpenGLESTextureCacheCreateTextureFromImage(kCFAllocatorDefault, coreVideoTextureCache, cameraFrame, NULL, GL_TEXTURE_2D, GL_RED_EXT, bufferWidth, bufferHeight, GL_RED_EXT, GL_UNSIGNED_BYTE, 0, &luminanceTextureRef);
                err = CVOpenGLESTextureCacheCreateTextureFromImage(kCFAllocatorDefault, [[GPUImageContext sharedImageProcessingContext] coreVideoTextureCache], cameraFrame, NULL, GL_TEXTURE_2D, GL_LUMINANCE, bufferWidth, bufferHeight, GL_LUMINANCE, GL_UNSIGNED_BYTE, 0, &luminanceTextureRef);
            }
            else
            {
                err = CVOpenGLESTextureCacheCreateTextureFromImage(kCFAllocatorDefault, [[GPUImageContext sharedImageProcessingContext] coreVideoTextureCache], cameraFrame, NULL, GL_TEXTURE_2D, GL_LUMINANCE, bufferWidth, bufferHeight, GL_LUMINANCE, GL_UNSIGNED_BYTE, 0, &luminanceTextureRef);
            }
            if (err)
            {
                NSLog(@"Error at CVOpenGLESTextureCacheCreateTextureFromImage %d", err);
            }
            
            luminanceTexture = CVOpenGLESTextureGetName(luminanceTextureRef);
            glBindTexture(GL_TEXTURE_2D, luminanceTexture);
            glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
            glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
            
            // UV-plane
            glActiveTexture(GL_TEXTURE5);
            if ([GPUImageContext deviceSupportsRedTextures])
            {
//                err = CVOpenGLESTextureCacheCreateTextureFromImage(kCFAllocatorDefault, coreVideoTextureCache, cameraFrame, NULL, GL_TEXTURE_2D, GL_RG_EXT, bufferWidth/2, bufferHeight/2, GL_RG_EXT, GL_UNSIGNED_BYTE, 1, &chrominanceTextureRef);
                err = CVOpenGLESTextureCacheCreateTextureFromImage(kCFAllocatorDefault, [[GPUImageContext sharedImageProcessingContext] coreVideoTextureCache], cameraFrame, NULL, GL_TEXTURE_2D, GL_LUMINANCE_ALPHA, bufferWidth/2, bufferHeight/2, GL_LUMINANCE_ALPHA, GL_UNSIGNED_BYTE, 1, &chrominanceTextureRef);
            }
            else
            {
                err = CVOpenGLESTextureCacheCreateTextureFromImage(kCFAllocatorDefault, [[GPUImageContext sharedImageProcessingContext] coreVideoTextureCache], cameraFrame, NULL, GL_TEXTURE_2D, GL_LUMINANCE_ALPHA, bufferWidth/2, bufferHeight/2, GL_LUMINANCE_ALPHA, GL_UNSIGNED_BYTE, 1, &chrominanceTextureRef);
            }
            if (err)
            {
                NSLog(@"Error at CVOpenGLESTextureCacheCreateTextureFromImage %d", err);
            }
            
            chrominanceTexture = CVOpenGLESTextureGetName(chrominanceTextureRef);
            glBindTexture(GL_TEXTURE_2D, chrominanceTexture);
            glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
            glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
            
//            if (!allTargetsWantMonochromeData)
//            {
                [self convertYUVToRGBOutput];
//            }

            int rotatedImageBufferWidth = bufferWidth, rotatedImageBufferHeight = bufferHeight;
            
            if (GPUImageRotationSwapsWidthAndHeight(internalRotation))
            {
                rotatedImageBufferWidth = bufferHeight;
                rotatedImageBufferHeight = bufferWidth;
            }
            
            [self updateTargetsForVideoCameraUsingCacheTextureAtWidth:rotatedImageBufferWidth height:rotatedImageBufferHeight time:currentTime];
            
            CVPixelBufferUnlockBaseAddress(cameraFrame, 0);
            CFRelease(luminanceTextureRef);
            CFRelease(chrominanceTextureRef);
        }

    }
}

- (void)convertYUVToRGBOutput;
{
    [GPUImageContext setActiveShaderProgram:yuvConversionProgram];

    int rotatedImageBufferWidth = imageBufferWidth, rotatedImageBufferHeight = imageBufferHeight;

    if (GPUImageRotationSwapsWidthAndHeight(internalRotation))
    {
        rotatedImageBufferWidth = imageBufferHeight;
        rotatedImageBufferHeight = imageBufferWidth;
    }

    outputFramebuffer = [[GPUImageContext sharedFramebufferCache] fetchFramebufferForSize:CGSizeMake(rotatedImageBufferWidth, rotatedImageBufferHeight) textureOptions:self.outputTextureOptions onlyTexture:NO];
    [outputFramebuffer activateFramebuffer];

    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    
    static const GLfloat squareVertices[] = {
        -1.0f, -1.0f,
        1.0f, -1.0f,
        -1.0f,  1.0f,
        1.0f,  1.0f,
    };
    
    glActiveTexture(GL_TEXTURE4);
    glBindTexture(GL_TEXTURE_2D, luminanceTexture);
    glUniform1i(yuvConversionLuminanceTextureUniform, 4);

    glActiveTexture(GL_TEXTURE5);
    glBindTexture(GL_TEXTURE_2D, chrominanceTexture);
    glUniform1i(yuvConversionChrominanceTextureUniform, 5);

    glUniformMatrix3fv(yuvConversionMatrixUniform, 1, GL_FALSE, _preferredConversion);

    glVertexAttribPointer(yuvConversionPositionAttribute, 2, GL_FLOAT, 0, 0, squareVertices);
    glVertexAttribPointer(yuvConversionTextureCoordinateAttribute, 2, GL_FLOAT, 0, 0, [GPUImageFilter textureCoordinatesForRotation:internalRotation]);
    
    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
    
    
    CGImageRef imageRef = [outputFramebuffer newCGImageFromFramebufferContents];
    UIImage *image = [UIImage imageWithCGImage:imageRef];
    NSLog(@"image:%@",image);
    
}

- (void)updateTargetsForVideoCameraUsingCacheTextureAtWidth:(int)bufferWidth height:(int)bufferHeight time:(CMTime)currentTime;
{
    // First, update all the framebuffers in the targets
    for (id<GPUImageInput> currentTarget in targets)
    {
//        if ([currentTarget enabled])
        {
            NSInteger indexOfObject = [targets indexOfObject:currentTarget];
            NSInteger textureIndexOfTarget = [[targetTextureIndices objectAtIndex:indexOfObject] integerValue];

//            if (currentTarget != self.targetToIgnoreForUpdates)
            {
                [currentTarget setInputRotation:outputRotation atIndex:textureIndexOfTarget];
                [currentTarget setInputSize:CGSizeMake(bufferWidth, bufferHeight) atIndex:textureIndexOfTarget];

//                if ([currentTarget wantsMonochromeInput] && captureAsYUV)
//                {
//                    [currentTarget setCurrentlyReceivingMonochromeInput:YES];
//                    // TODO: Replace optimization for monochrome output
//                    [currentTarget setInputFramebuffer:outputFramebuffer atIndex:textureIndexOfTarget];
//                }
//                else
                {
//                    [currentTarget setCurrentlyReceivingMonochromeInput:NO];
                    [currentTarget setInputFramebuffer:outputFramebuffer atIndex:textureIndexOfTarget];
                }
            }
//            else
//            {
//                [currentTarget setInputRotation:outputRotation atIndex:textureIndexOfTarget];
//                [currentTarget setInputFramebuffer:outputFramebuffer atIndex:textureIndexOfTarget];
//            }
        }
    }
    
    // Then release our hold on the local framebuffer to send it back to the cache as soon as it's no longer needed
    [outputFramebuffer unlock];
    outputFramebuffer = nil;
    
    // Finally, trigger rendering as needed
    for (id<GPUImageInput> currentTarget in targets)
    {
//        if ([currentTarget enabled])
        {
            NSInteger indexOfObject = [targets indexOfObject:currentTarget];
            NSInteger textureIndexOfTarget = [[targetTextureIndices objectAtIndex:indexOfObject] integerValue];
//
//            if (currentTarget != self.targetToIgnoreForUpdates)
            {
                [currentTarget newFrameReadyAtTime:currentTime atIndex:textureIndexOfTarget];
            }
        }
    }
}

- (void)startCameraCapture;
{
    if (![_captureSession isRunning])
    {
//        startingCaptureTime = [NSDate date];
        [_captureSession startRunning];
    };
}

- (void)stopCameraCapture;
{
    if ([_captureSession isRunning])
    {
        [_captureSession stopRunning];
    }
}

- (void)pauseCameraCapture;
{
//    capturePaused = YES;
}

- (void)resumeCameraCapture;
{
//    capturePaused = NO;
}

- (AVCaptureDevicePosition)cameraPosition
{
    return [[videoInput device] position];
}

- (void)setOutputImageOrientation:(UIInterfaceOrientation)newValue;
{
    _outputImageOrientation = newValue;
    [self updateOrientationSendToTargets];
}

- (void)updateOrientationSendToTargets;
{
    runSynchronouslyOnVideoProcessingQueue(^{
        
        //    From the iOS 5.0 release notes:
        //    In previous iOS versions, the front-facing camera would always deliver buffers in AVCaptureVideoOrientationLandscapeLeft and the back-facing camera would always deliver buffers in AVCaptureVideoOrientationLandscapeRight.
        
        if (captureAsYUV && [GPUImageContext supportsFastTextureUpload])
        {
            outputRotation = kGPUImageNoRotation;
            if ([self cameraPosition] == AVCaptureDevicePositionBack)
            {
//                if (_horizontallyMirrorRearFacingCamera)
//                {
//                    switch(_outputImageOrientation)
//                    {
//                        case UIInterfaceOrientationPortrait:internalRotation = kGPUImageRotateRightFlipVertical; break;
//                        case UIInterfaceOrientationPortraitUpsideDown:internalRotation = kGPUImageRotate180; break;
//                        case UIInterfaceOrientationLandscapeLeft:internalRotation = kGPUImageFlipHorizonal; break;
//                        case UIInterfaceOrientationLandscapeRight:internalRotation = kGPUImageFlipVertical; break;
//                        default:internalRotation = kGPUImageNoRotation;
//                    }
//                }
//                else
                {
                    switch(_outputImageOrientation)
                    {
                        case UIInterfaceOrientationPortrait:internalRotation = kGPUImageRotateRight; break;
                        case UIInterfaceOrientationPortraitUpsideDown:internalRotation = kGPUImageRotateLeft; break;
                        case UIInterfaceOrientationLandscapeLeft:internalRotation = kGPUImageRotate180; break;
                        case UIInterfaceOrientationLandscapeRight:internalRotation = kGPUImageNoRotation; break;
                        default:internalRotation = kGPUImageNoRotation;
                    }
                }
            }
            else
            {
//                if (_horizontallyMirrorFrontFacingCamera)
//                {
//                    switch(_outputImageOrientation)
//                    {
//                        case UIInterfaceOrientationPortrait:internalRotation = kGPUImageRotateRightFlipVertical; break;
//                        case UIInterfaceOrientationPortraitUpsideDown:internalRotation = kGPUImageRotateRightFlipHorizontal; break;
//                        case UIInterfaceOrientationLandscapeLeft:internalRotation = kGPUImageFlipHorizonal; break;
//                        case UIInterfaceOrientationLandscapeRight:internalRotation = kGPUImageFlipVertical; break;
//                        default:internalRotation = kGPUImageNoRotation;
//                   }
//                }
//                else
                {
                    switch(_outputImageOrientation)
                    {
                        case UIInterfaceOrientationPortrait:internalRotation = kGPUImageRotateRight; break;
                        case UIInterfaceOrientationPortraitUpsideDown:internalRotation = kGPUImageRotateLeft; break;
                        case UIInterfaceOrientationLandscapeLeft:internalRotation = kGPUImageNoRotation; break;
                        case UIInterfaceOrientationLandscapeRight:internalRotation = kGPUImageRotate180; break;
                        default:internalRotation = kGPUImageNoRotation;
                    }
                }
            }
        }
//        else
//        {
//            if ([self cameraPosition] == AVCaptureDevicePositionBack)
//            {
//                if (_horizontallyMirrorRearFacingCamera)
//                {
//                    switch(_outputImageOrientation)
//                    {
//                        case UIInterfaceOrientationPortrait:outputRotation = kGPUImageRotateRightFlipVertical; break;
//                        case UIInterfaceOrientationPortraitUpsideDown:outputRotation = kGPUImageRotate180; break;
//                        case UIInterfaceOrientationLandscapeLeft:outputRotation = kGPUImageFlipHorizonal; break;
//                        case UIInterfaceOrientationLandscapeRight:outputRotation = kGPUImageFlipVertical; break;
//                        default:outputRotation = kGPUImageNoRotation;
//                    }
//                }
//                else
//                {
//                    switch(_outputImageOrientation)
//                    {
//                        case UIInterfaceOrientationPortrait:outputRotation = kGPUImageRotateRight; break;
//                        case UIInterfaceOrientationPortraitUpsideDown:outputRotation = kGPUImageRotateLeft; break;
//                        case UIInterfaceOrientationLandscapeLeft:outputRotation = kGPUImageRotate180; break;
//                        case UIInterfaceOrientationLandscapeRight:outputRotation = kGPUImageNoRotation; break;
//                        default:outputRotation = kGPUImageNoRotation;
//                    }
//                }
//            }
//            else
//            {
//                if (_horizontallyMirrorFrontFacingCamera)
//                {
//                    switch(_outputImageOrientation)
//                    {
//                        case UIInterfaceOrientationPortrait:outputRotation = kGPUImageRotateRightFlipVertical; break;
//                        case UIInterfaceOrientationPortraitUpsideDown:outputRotation = kGPUImageRotateRightFlipHorizontal; break;
//                        case UIInterfaceOrientationLandscapeLeft:outputRotation = kGPUImageFlipHorizonal; break;
//                        case UIInterfaceOrientationLandscapeRight:outputRotation = kGPUImageFlipVertical; break;
//                        default:outputRotation = kGPUImageNoRotation;
//                    }
//                }
//                else
//                {
//                    switch(_outputImageOrientation)
//                    {
//                        case UIInterfaceOrientationPortrait:outputRotation = kGPUImageRotateRight; break;
//                        case UIInterfaceOrientationPortraitUpsideDown:outputRotation = kGPUImageRotateLeft; break;
//                        case UIInterfaceOrientationLandscapeLeft:outputRotation = kGPUImageNoRotation; break;
//                        case UIInterfaceOrientationLandscapeRight:outputRotation = kGPUImageRotate180; break;
//                        default:outputRotation = kGPUImageNoRotation;
//                    }
//                }
//            }
//        }
        
        for (id<GPUImageInput> currentTarget in targets)
        {
            NSInteger indexOfObject = [targets indexOfObject:currentTarget];
            [currentTarget setInputRotation:outputRotation atIndex:[[targetTextureIndices objectAtIndex:indexOfObject] integerValue]];
        }
    });
}
@end

//
//  GPUImageVideoCamera.h
//  opengles01
//
//  Created by libb on 2020/12/27.
//  Copyright © 2020 Rhythmic Fistman. All rights reserved.
//

#import <Foundation/Foundation.h>
#import <AVFoundation/AVFoundation.h>
#import "GPUImageContext.h"
NS_ASSUME_NONNULL_BEGIN

@interface GPUImageVideoCamera : GPUImageOutput
{
    AVCaptureSession *_captureSession;
    AVCaptureDevice *_inputCamera;
    AVCaptureDevice *_microphone;
    AVCaptureDeviceInput *videoInput;
    AVCaptureVideoDataOutput *videoOutput;
    
    dispatch_queue_t cameraProcessingQueue, audioProcessingQueue;
    
    GPUImageRotationMode outputRotation, internalRotation;
    
    BOOL captureAsYUV;
    
    int imageBufferWidth, imageBufferHeight;
    GLuint luminanceTexture, chrominanceTexture;
    
    BOOL isFullYUVRange;
}

/// The AVCaptureSession used to capture from the camera
@property(readonly, retain, nonatomic) AVCaptureSession *captureSession;

/// This enables the capture session preset to be changed on the fly
@property (readwrite, nonatomic, copy) NSString *captureSessionPreset;

/// This determines the rotation applied to the output image, based on the source material
@property(readwrite, nonatomic) UIInterfaceOrientation outputImageOrientation;

- (id)initWithSessionPreset:(NSString *)sessionPreset cameraPosition:(AVCaptureDevicePosition)cameraPosition;

/** Start camera capturing
 */
- (void)startCameraCapture;

/** Stop camera capturing
 */
- (void)stopCameraCapture;

/** Pause camera capturing
 */
- (void)pauseCameraCapture;

/** Resume camera capturing
 */
- (void)resumeCameraCapture;


@end

NS_ASSUME_NONNULL_END

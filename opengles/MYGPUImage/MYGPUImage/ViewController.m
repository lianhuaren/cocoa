//
//  ViewController.m
//  RPScreenRecorder
//
//  Created by Rhythmic Fistman on 20/6/18.
//  Copyright © 2018 Rhythmic Fistman. All rights reserved.
//

// https://stackoverflow.com/questions/50935432/rpscreenrecorder-startcapturewithhandler-not-returning-microphone-sound-in-samp

#import "ViewController.h"
#import <ReplayKit/ReplayKit.h>
#import <AVFoundation/AVFoundation.h>
#import "LFLiveVideoConfiguration.h"
#import "GPUImageVideoCamera.h"
#import "GPUImageView.h"
@interface ViewController ()

@property (nonatomic, strong) GPUImageVideoCamera *videoCamera;
@property (nonatomic, strong) LFLiveVideoConfiguration *configuration;
@property (nonatomic, strong) GPUImageView *gpuImageView;
@end

@implementation ViewController


- (void)viewDidLoad {
    [super viewDidLoad];
  
    [self setPreView:self.view];
    
    /***   默认分辨率368 ＊ 640  音频：44.1 iphone6以上48  双声道  方向竖屏 ***/
    LFLiveVideoConfiguration *videoConfiguration = [LFLiveVideoConfiguration new];
    videoConfiguration.videoSize = CGSizeMake(360, 640);
    videoConfiguration.videoBitRate = 800*1024;
    videoConfiguration.videoMaxBitRate = 1000*1024;
    videoConfiguration.videoMinBitRate = 500*1024;
    videoConfiguration.videoFrameRate = 24;
    videoConfiguration.videoMaxKeyframeInterval = 48;
    videoConfiguration.outputImageOrientation = UIInterfaceOrientationPortrait;
    videoConfiguration.autorotate = NO;
    videoConfiguration.sessionPreset = LFCaptureSessionPreset720x1280;
    
    self.configuration = videoConfiguration;
    
    _videoCamera = [[GPUImageVideoCamera alloc] initWithSessionPreset:_configuration.avSessionPreset cameraPosition:AVCaptureDevicePositionFront];
    _videoCamera.outputImageOrientation = _configuration.outputImageOrientation;
//    _videoCamera.horizontallyMirrorFrontFacingCamera = NO;
//    _videoCamera.horizontallyMirrorRearFacingCamera = NO;
    
    
    [_videoCamera addTarget:self.gpuImageView];
    
    [_videoCamera startCameraCapture];
}

//- (GPUImageVideoCamera *)videoCamera{
//    if(!_videoCamera){
//        _videoCamera = [[GPUImageVideoCamera alloc] initWithSessionPreset:_configuration.avSessionPreset cameraPosition:AVCaptureDevicePositionFront];
//        _videoCamera.outputImageOrientation = _configuration.outputImageOrientation;
//        _videoCamera.horizontallyMirrorFrontFacingCamera = NO;
//        _videoCamera.horizontallyMirrorRearFacingCamera = NO;
//        _videoCamera.frameRate = (int32_t)_configuration.videoFrameRate;
//    }
//    return _videoCamera;
//}
- (void)setPreView:(UIView *)preView {
    if (self.gpuImageView.superview) [self.gpuImageView removeFromSuperview];
    [preView insertSubview:self.gpuImageView atIndex:0];
    self.gpuImageView.frame = CGRectMake(0, 0, preView.frame.size.width, preView.frame.size.height);
}

- (GPUImageView *)gpuImageView{
    if(!_gpuImageView){
        _gpuImageView = [[GPUImageView alloc] initWithFrame:[UIScreen mainScreen].bounds];
        [_gpuImageView setFillMode:kGPUImageFillModePreserveAspectRatioAndFill];
        [_gpuImageView setAutoresizingMask:UIViewAutoresizingFlexibleWidth | UIViewAutoresizingFlexibleHeight];
    }
    return _gpuImageView;
}



@end

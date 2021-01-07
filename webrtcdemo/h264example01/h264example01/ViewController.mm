//
//  ViewController.m
//  h264example01
//
//  Created by libb on 2021/1/7.
//

#import "ViewController.h"
#import <ReplayKit/ReplayKit.h>
#include "video_coding/codecs/h264/include/h264.h"
#include "video_coding/main/source/codec_database.h"
#include "libyuv/convert.h"
#include "base/checks.h"
#include "base/logging.h"
#define kIntBitRate 500
#define kVideoWidth 640
#define kVideoHeight 480
#define kFrameRate 15

using namespace webrtc;

rtc::scoped_refptr<webrtc::VideoFrameBuffer> VideoFrameBufferForPixelBuffermy(
    CVPixelBufferRef pixel_buffer) {
  RTC_DCHECK(pixel_buffer);
  RTC_DCHECK(CVPixelBufferGetPixelFormatType(pixel_buffer) ==
             kCVPixelFormatType_420YpCbCr8BiPlanarFullRange);
  size_t width = CVPixelBufferGetWidthOfPlane(pixel_buffer, 0);
  size_t height = CVPixelBufferGetHeightOfPlane(pixel_buffer, 0);
  // TODO(tkchin): Use a frame buffer pool.
  rtc::scoped_refptr<webrtc::VideoFrameBuffer> buffer =
      new rtc::RefCountedObject<webrtc::I420Buffer>(width, height);
  CVPixelBufferLockBaseAddress(pixel_buffer, kCVPixelBufferLock_ReadOnly);
  const uint8_t* src_y = reinterpret_cast<const uint8_t*>(
      CVPixelBufferGetBaseAddressOfPlane(pixel_buffer, 0));
  int src_y_stride = CVPixelBufferGetBytesPerRowOfPlane(pixel_buffer, 0);
  const uint8_t* src_uv = reinterpret_cast<const uint8_t*>(
      CVPixelBufferGetBaseAddressOfPlane(pixel_buffer, 1));
  int src_uv_stride = CVPixelBufferGetBytesPerRowOfPlane(pixel_buffer, 1);
  int ret = libyuv::NV12ToI420(
      src_y, src_y_stride, src_uv, src_uv_stride,
      buffer->MutableData(webrtc::kYPlane), buffer->stride(webrtc::kYPlane),
      buffer->MutableData(webrtc::kUPlane), buffer->stride(webrtc::kUPlane),
      buffer->MutableData(webrtc::kVPlane), buffer->stride(webrtc::kVPlane),
      width, height);
  CVPixelBufferUnlockBaseAddress(pixel_buffer, kCVPixelBufferLock_ReadOnly);
  if (ret) {
    LOG(LS_ERROR) << "Error converting NV12 to I420: " << ret;
    return nullptr;
  }
  return buffer;
}

@interface ViewController ()
{
    H264Encoder* encoder_;
//    VCMGenericEncoder* ptr_encoder_;
    VideoEncoderRateObserver* encoder_rate_observer_;
    
    
}
@end

@implementation ViewController

- (void)viewDidLoad {
    [super viewDidLoad];
    // Do any additional setup after loading the view.
    [self setUpUI];
    
    
    rtc::scoped_ptr<uint8_t[]> new_data(new uint8_t[10]);

}

- (void)setUpUI
{
    UIView *testView = ({
        UIView *view = [[UIView alloc] initWithFrame:CGRectMake(100, 0, 100, 100)];
        [view setBackgroundColor:[UIColor purpleColor]];
        view;
    });
    [self.view addSubview:testView];

    /*
     这个动画会让直播一直有视频帧
     动画类型不限，只要屏幕是变化的就会有视频帧
     */
    [testView.layer removeAllAnimations];
    CABasicAnimation *rA = [CABasicAnimation animationWithKeyPath:@"transform.rotation.z"];
    rA.duration = 3.0;
    rA.toValue = [NSNumber numberWithFloat:M_PI * 2];
    rA.repeatCount = MAXFLOAT;
    rA.removedOnCompletion = NO;
    [testView.layer addAnimation:rA forKey:@""];
    
    UIButton *startBtn = [UIButton buttonWithType:UIButtonTypeCustom];
    [self.view addSubview:startBtn];
    [startBtn setTitle:@"开始" forState:UIControlStateNormal];
    [startBtn addTarget:self action:@selector(startRecord) forControlEvents:UIControlEventTouchUpInside];
    startBtn.frame = CGRectMake(30, 300, 80, 40);
    [startBtn setTitleColor:[UIColor blackColor] forState:UIControlStateNormal];
    startBtn.layer.borderWidth = 0.5;
    startBtn.layer.borderColor = [UIColor blackColor].CGColor;
    
    UIButton *stopBtn = [UIButton buttonWithType:UIButtonTypeCustom];
    [self.view addSubview:stopBtn];
    [stopBtn setTitle:@"结束" forState:UIControlStateNormal];
    [stopBtn addTarget:self action:@selector(stopRecord) forControlEvents:UIControlEventTouchUpInside];
    stopBtn.frame = CGRectMake(140, 300, 80, 40);
    [stopBtn setTitleColor:[UIColor blackColor] forState:UIControlStateNormal];
    stopBtn.layer.borderWidth = 0.5;
    stopBtn.layer.borderColor = [UIColor blackColor].CGColor;
    

}

-(void)initEncoderWithWidth:(int)width height:(int)height
{
    //    encoder_rate_observer_ = NULL;
    //    ptr_encoder_ = new VCMGenericEncoder(H264Encoder::Create(),
    //                                 encoder_rate_observer_,
    //                          false);
        
        encoder_ = H264Encoder::Create();
        
        int numberOfCores = 0;
        size_t maxPayloadSize = 0;
        
//        CGSize s = [[UIScreen mainScreen] bounds].size;
//        int width = (int)(s.width) / 4 * 4;
//        int height = (int)(s.height) / 4 * 4;
        
        
        //Create codec
        webrtc::VideoCodec codec;
    //    vieData.codec->GetCodec(codecType, codec);
        codec.codecType = kVideoCodecH264;
        codec.startBitrate = kIntBitRate;
        codec.maxBitrate = 600;
        codec.width = width;
        codec.height = height;
        codec.maxFramerate = kFrameRate;
        
        if (encoder_->InitEncode(&codec, numberOfCores, maxPayloadSize) != 0) {
    //      LOG(LS_ERROR) << "Failed to initialize the encoder associated with "
    //                       "payload name: " << settings->plName;
    //      return -1;
        }
        
    //    if (ptr_encoder_->InitEncode(&codec, number_of_cores_,
    //                                 max_payload_size_) < 0) {
    //      LOG(LS_ERROR) << "Failed to initialize video encoder.";
    //      DeleteEncoder();
    //      return false;
    //    }
    //    else if (ptr_encoder_->RegisterEncodeCallback(encoded_frame_callback) < 0) {
    //      LOG(LS_ERROR) << "Failed to register encoded-frame callback.";
    //      DeleteEncoder();
    //      return false;
    //    }
}

- (void)processSampleBuffer:(CMSampleBufferRef)sampleBuffer {
//    CFStringRef RPVideoSampleOrientationKeyRef = (__bridge CFStringRef)RPVideoSampleOrientationKey;
//    NSNumber *orientation = (__bridge NSNumber *)CMGetAttachment(sampleBuffer, RPVideoSampleOrientationKeyRef, NULL);
//

    
    CVImageBufferRef image_buffer = CMSampleBufferGetImageBuffer(sampleBuffer);

    size_t width = CVPixelBufferGetWidthOfPlane(image_buffer, 0);
    size_t height = CVPixelBufferGetHeightOfPlane(image_buffer, 0);
    NSLog(@"width:%d height:%d",width, height);
    
    if (!encoder_) {
        [self initEncoderWithWidth:width height:height];
    }
    
    rtc::scoped_refptr<webrtc::VideoFrameBuffer> buffer =
    VideoFrameBufferForPixelBuffermy(image_buffer);
    webrtc::VideoFrame frame(buffer, 0, 0,
                                     webrtc::kVideoRotation_0);
    
//    std::vector<FrameType> _nextFrameTypes(1, kVideoFrameDelta);
    std::vector<VideoFrameType> video_frame_types(1,
                                                  kDeltaFrame);
    
    CodecSpecificInfo codecSpecificInfo;
    int32_t result =
        encoder_->Encode(frame, &codecSpecificInfo, &video_frame_types);
    
//    int32_t ret = encoder_->Encode(frame, &codecSpecificInfo, video_frame_types);
//    if (ret < 0) {
//      LOG(LS_ERROR) << "Failed to encode frame. Error code: " << ret;
//      return ret;
//    }
    
}


- (void)startRecord
{
    
    
    RPScreenRecorder *recorder = [RPScreenRecorder sharedRecorder];
    recorder.microphoneEnabled = NO;
    
    [recorder startCaptureWithHandler:^(CMSampleBufferRef sampleBuffer, RPSampleBufferType bufferType, NSError* error) {
        if (error) {
            NSLog(@"Capture %@, %li, %@", sampleBuffer, (long)bufferType, error);
            return;
        }

//TODO 编码并处理

       if (RPSampleBufferTypeVideo == bufferType) {
           
           CMTime time = kCMTimeZero;
           time = CMSampleBufferGetPresentationTimeStamp(sampleBuffer);
           float pts = (float)time.value / time.timescale;
           
//           static int frameCount = 0;
//           frameCount++;
//           CMTime presentationTimeStamp = CMTimeMake(frameCount, (int32_t)m_fps);
//           CMTime duration = CMTimeMake(1, (int32_t)m_fps);
//
//           float pts = (float)presentationTimeStamp.value / presentationTimeStamp.timescale;
//
//
           NSLog(@"===pts:%.2f",pts);
           


           [self processSampleBuffer:sampleBuffer];

        } else if (RPSampleBufferTypeAudioApp == bufferType) {

        } else if (RPSampleBufferTypeAudioMic == bufferType) {

        }


    } completionHandler:^(NSError* error) {
        NSLog(@"startCapture: %@", error);
    }];
    
//    NSLog(@"开始录制:%d",ret);
}

- (void)stopRecord
{

    
    RPScreenRecorder *recorder = [RPScreenRecorder sharedRecorder];
    [recorder stopCaptureWithHandler:^(NSError * _Nullable error) {
        NSLog(@"stopCaptureWithHandler: %@", error);
    }];
    
    NSLog(@"结束录制");
}
@end

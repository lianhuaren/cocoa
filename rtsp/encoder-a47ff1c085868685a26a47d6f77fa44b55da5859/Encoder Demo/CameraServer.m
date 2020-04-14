//
//  CameraServer.m
//  Encoder Demo
//
//  Created by Geraint Davies on 19/02/2013.
//  Copyright (c) 2013 GDCL http://www.gdcl.co.uk/license.htm
//

#import "CameraServer.h"
#import "AVEncoder.h"
#import "RTSPServer.h"
#import <CoreImage/CoreImage.h>

static CameraServer* theServer;

@interface CameraServer  () <AVCaptureVideoDataOutputSampleBufferDelegate>
{
    AVCaptureSession* _session;
    AVCaptureVideoPreviewLayer* _preview;
    AVCaptureVideoDataOutput* _output;
    dispatch_queue_t _captureQueue;
    
    AVEncoder* _encoder;
    
    RTSPServer* _rtsp;
}
@property (nonatomic, strong) CIContext *ciContext;

@end


@implementation CameraServer

+ (void) initialize
{
    // test recommended to avoid duplicate init via subclass
    if (self == [CameraServer class])
    {
        theServer = [[CameraServer alloc] init];
    }
}

+ (CameraServer*) server
{
    return theServer;
}

- (void) startup
{
    if (_session == nil)
    {
        NSLog(@"Starting up server");
        
        // create capture device with video input
        _session = [[AVCaptureSession alloc] init];
        AVCaptureDevice* dev = [AVCaptureDevice defaultDeviceWithMediaType:AVMediaTypeVideo];
        AVCaptureDeviceInput* input = [AVCaptureDeviceInput deviceInputWithDevice:dev error:nil];
        [_session addInput:input];
        
        // create an output for YUV output with self as delegate
        _captureQueue = dispatch_queue_create("uk.co.gdcl.avencoder.capture", DISPATCH_QUEUE_SERIAL);
        _output = [[AVCaptureVideoDataOutput alloc] init];
        [_output setSampleBufferDelegate:self queue:_captureQueue];
        NSDictionary* setcapSettings = [NSDictionary dictionaryWithObjectsAndKeys:
                                        [NSNumber numberWithInt:kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange], kCVPixelBufferPixelFormatTypeKey,
                                        nil];
        _output.videoSettings = setcapSettings;
        [_session addOutput:_output];
        
        // create an encoder
        _encoder = [AVEncoder encoderForHeight:480 andWidth:720];
        [_encoder encodeWithBlock:^int(NSArray* data, double pts) {
            if (_rtsp != nil)
            {
                _rtsp.bitrate = _encoder.bitspersecond;
                [_rtsp onVideoData:data time:pts];
            }
            return 0;
        } onParams:^int(NSData *data) {
            _rtsp = [RTSPServer setupListener:data];
            return 0;
        }];
        
        // start capture and a preview layer
        [_session startRunning];
        
        
        _preview = [AVCaptureVideoPreviewLayer layerWithSession:_session];
        _preview.videoGravity = AVLayerVideoGravityResizeAspectFill;
    }
}

- (void) captureOutput:(AVCaptureOutput *)captureOutput didOutputSampleBuffer:(CMSampleBufferRef)sampleBuffer fromConnection:(AVCaptureConnection *)connection
{
    [self dealWithSampleBuffer:sampleBuffer];
    
    // pass frame to encoder
    [_encoder encodeFrame:sampleBuffer];
}

- (void)dealWithSampleBuffer:(CMSampleBufferRef)buffer {

    CVPixelBufferRef pixelBuffer = CMSampleBufferGetImageBuffer(buffer);
    CIImage *ciimage = [CIImage imageWithCVPixelBuffer:pixelBuffer];
    size_t width = CVPixelBufferGetWidth(pixelBuffer);
    size_t height = CVPixelBufferGetHeight(pixelBuffer);

    NSLog(@"pixelBuffer width:%d height:%d",width,height);
    
    CGFloat widthScale = width/720.0;
    CGFloat heightScale = height/1280.0;
    CGFloat realWidthScale = 1;
    CGFloat realHeightScale = 1;
    
    if (widthScale > 1 || heightScale > 1) {
        if (widthScale < heightScale) {
            realHeightScale = 1280.0/height;
            CGFloat nowWidth = width * 1280 / height;
            height = 1280;
            realWidthScale = nowWidth/width;
            width = nowWidth;
        } else {
            realWidthScale = 720.0/width;
            CGFloat nowHeight = 720 * height / width;
            width = 720;
            realHeightScale = nowHeight/height;
            height = nowHeight;
        }
    }
    
    
    {
        _ciContext = [CIContext contextWithOptions:nil];
        
        CIImage *newImage = [ciimage imageByApplyingTransform:CGAffineTransformMakeScale(realWidthScale, realHeightScale)];
//        UIImage *tmpImage = [self imageWithColor:[UIColor redColor] AndRect:CGRectMake(0, 0, width, height)];
//        CIImage *newImage = [CIImage imageWithCGImage:tmpImage.CGImage];
        
        CVPixelBufferLockBaseAddress(pixelBuffer, 0);
        CVPixelBufferRef newPixcelBuffer = nil;
        CVPixelBufferCreate(kCFAllocatorDefault, width, height, kCVPixelFormatType_32BGRA, nil, &newPixcelBuffer);
        [_ciContext render:newImage toCVPixelBuffer:newPixcelBuffer];
        CVPixelBufferUnlockBaseAddress(pixelBuffer, 0);
//        [self.videoEncoder encodeVideoData:newPixcelBuffer timeStamp:(CACurrentMediaTime()*1000)];
        
        size_t newWidth = CVPixelBufferGetWidth(newPixcelBuffer);
        size_t newHeight = CVPixelBufferGetHeight(newPixcelBuffer);
        NSLog(@"newPixcelBuffer width:%d height:%d",newWidth,newHeight);
        
        UIImage* sampleImage = [self imageFromSamplePlanerPixelBuffer:newPixcelBuffer];
        
        CVPixelBufferRelease(newPixcelBuffer);
    }
}

- (UIImage *)imageWithColor:(UIColor *)color AndRect:(CGRect)rect{

    UIGraphicsBeginImageContext(rect.size);

    CGContextRef context = UIGraphicsGetCurrentContext();

    

    CGContextSetFillColorWithColor(context, [color CGColor]);

    CGContextFillRect(context, rect);

    

    UIImage *image = UIGraphicsGetImageFromCurrentImageContext();

    UIGraphicsEndImageContext();

    

    return image;

}

- (UIImage *) imageFromSamplePlanerPixelBuffer:(CVPixelBufferRef)imageBuffer{
    @autoreleasepool {
//        // Get a CMSampleBuffer's Core Video image buffer for the media data
//        CVImageBufferRef imageBuffer = CMSampleBufferGetImageBuffer(sampleBuffer);
        // Lock the base address of the pixel buffer
        CVPixelBufferLockBaseAddress(imageBuffer, 0);
        
        // Get the number of bytes per row for the plane pixel buffer
        void *baseAddress = CVPixelBufferGetBaseAddressOfPlane(imageBuffer, 0);
        
        // Get the number of bytes per row for the plane pixel buffer
        size_t bytesPerRow = CVPixelBufferGetBytesPerRowOfPlane(imageBuffer,0);
        // Get the pixel buffer width and height
        size_t width = CVPixelBufferGetWidth(imageBuffer);
        size_t height = CVPixelBufferGetHeight(imageBuffer);
        

        size_t size = CVPixelBufferGetDataSize(imageBuffer);

        OSType type = CVPixelBufferGetPixelFormatType(imageBuffer);
        
        NSLog(@"buffer type:%d size:%d",type,size);
        
        static int i=0;
        i++;
        if (i<4) {
            


            int len = (int)width * height *4;
            uint8_t *rgb_frame = (uint8_t *)malloc(len);
            
            for(int y = 0; y < height; y++) {
                 uint8_t *yBufferLine = &baseAddress[y * bytesPerRow];
                for(int x = 0; x < bytesPerRow; x++) {
                
                    rgb_frame[x+y * bytesPerRow] = yBufferLine[x];
                }
            }
            
            NSString *path2 = [self getHome2Path];
            const char *resultCString2 = NULL;
            if ([path2 canBeConvertedToEncoding:NSUTF8StringEncoding]) {
                resultCString2 = [path2 cStringUsingEncoding:NSUTF8StringEncoding];
            }
            
            unsigned char *buffer = (unsigned char *)rgb_frame;

            FILE* fpyuv = fopen(resultCString2, "wb");
            for (int i = 0; i < len; i ++) {
                fwrite(buffer, 1, 1, fpyuv);
                buffer ++;
            }
            fclose(fpyuv);
            
            free(rgb_frame);
        }
        
        
        // Create a device-dependent RGB color space
        CGColorSpaceRef colorSpace = CGColorSpaceCreateDeviceRGB();
        
        // Create a bitmap graphics context with the sample buffer data
        CGContextRef context = CGBitmapContextCreate(baseAddress, width, height, 8,
                                                     bytesPerRow, colorSpace, kCGImageAlphaNoneSkipFirst | kCGBitmapByteOrder32Little);
        // Create a Quartz image from the pixel data in the bitmap graphics context
        CGImageRef quartzImage = CGBitmapContextCreateImage(context);
        // Unlock the pixel buffer
        CVPixelBufferUnlockBaseAddress(imageBuffer,0);
        
        // Free up the context and color space
        CGContextRelease(context);
        CGColorSpaceRelease(colorSpace);
        
        // Create an image object from the Quartz image
        UIImage *image = [UIImage imageWithCGImage:quartzImage];
        
        // Release the Quartz image
        CGImageRelease(quartzImage);
        return (image);
    }
}

- (void) shutdown
{
    NSLog(@"shutting down server");
    if (_session)
    {
        [_session stopRunning];
        _session = nil;
    }
    if (_rtsp)
    {
        [_rtsp shutdownServer];
    }
    if (_encoder)
    {
        [ _encoder shutdown];
    }
}

- (NSString*) getURL
{
    NSString* ipaddr = [RTSPServer getIPAddress];
    NSString* url = [NSString stringWithFormat:@"rtsp://%@/", ipaddr];
    NSLog(@"------url:%@",url);
    return url;
}

- (AVCaptureVideoPreviewLayer*) getPreviewLayer
{
    return _preview;
}

- (NSString *)getHome2Path{
    NSString *path = NSHomeDirectory();
    
    NSArray *paths = NSSearchPathForDirectoriesInDomains(NSDocumentDirectory, NSUserDomainMask, YES);
    NSString *docDir = [paths objectAtIndex:0];
    static int i=0;
    return [NSString stringWithFormat:@"%@/%d.rgb",docDir,i++];
}

@end

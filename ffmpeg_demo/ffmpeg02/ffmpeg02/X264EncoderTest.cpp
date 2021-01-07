//
//  X264EncoderTest.cpp
//  ffmpeg02
//
//  Created by libb on 2020/12/30.
//

#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

#include <libavformat/avformat.h>

#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libswscale/swscale.h>
#include <libavfilter/avfilter.h>
#include <libavutil/imgutils.h>

#ifdef __cplusplus
};
#endif

typedef unsigned char                   UInt8;
typedef signed char                     SInt8;
typedef unsigned short                  UInt16;
typedef signed short                    SInt16;


AVCodecContext *pCodecCtx;
AVCodec *pCodec;
AVPacket packet;
AVFrame *pFrame;

int pictureSize;
int frameCounter;
int frameWidth;
int frameHeight;

FILE *fp;
int enabledWriteVideoFile;

void setupEncoderWithWidth(int width, int height) {
    avcodec_register_all();
    avcodec_register(NULL);
    
    frameCounter = 0;
    frameWidth = width;//self.videoSize.width;
    frameHeight = height;//self.videoSize.height;
    
    // Param that must set
    pCodecCtx = avcodec_alloc_context3(pCodec);
    pCodecCtx->codec_id = AV_CODEC_ID_H264;
    pCodecCtx->codec_type = AVMEDIA_TYPE_VIDEO;
    pCodecCtx->pix_fmt = AV_PIX_FMT_YUV420P;
    pCodecCtx->width = frameWidth;
    pCodecCtx->height = frameHeight;
    pCodecCtx->time_base.num = 1;
    pCodecCtx->time_base.den = 25;//self.frameRate;
    pCodecCtx->bit_rate = 1024*1000;//self.bitrate;
    pCodecCtx->gop_size =25;// self.maxKeyframeInterval;
    pCodecCtx->qmin = 10;
    pCodecCtx->qmax = 51;
    
    AVDictionary *param = NULL;
    
    if(pCodecCtx->codec_id == AV_CODEC_ID_H264) {
        av_dict_set(&param, "preset", "slow", 0);
        av_dict_set(&param, "tune", "zerolatency", 0);
    }
    
    pCodec = avcodec_find_encoder(pCodecCtx->codec_id);
    
    if (!pCodec) {
//        NSLog(@"Can not find encoder!");
    }
    
    if (avcodec_open2(pCodecCtx, pCodec, &param) < 0) {
//        NSLog(@"Failed to open encoder!");
    }
    
    pFrame = av_frame_alloc();
    pFrame->width = frameWidth;
    pFrame->height = frameHeight;
    pFrame->format = AV_PIX_FMT_YUV420P;
    
    avpicture_fill((AVPicture *)pFrame, NULL, pCodecCtx->pix_fmt, pCodecCtx->width, pCodecCtx->height);
    pictureSize = avpicture_get_size(pCodecCtx->pix_fmt, pCodecCtx->width, pCodecCtx->height);
    av_new_packet(&packet, pictureSize);
    
    
//    NSString *path = [self GetFilePathByfileName:@"IOSCamDemo.h264"];
//    NSLog(@"%@", path);
    fp = fopen("IOSCamDemo.h264", "wb");
}

void encoding() {
//    CVPixelBufferRef pixelBuffer = CMSampleBufferGetImageBuffer(sample);
//
//
////    [self savePixelBufferYUV:pixelBuffer];
//
//    CVPixelBufferLockBaseAddress(pixelBuffer, 0);
//
//    // 获取CVImageBufferRef中的y数据
//    UInt8 *pY = (UInt8 *)CVPixelBufferGetBaseAddressOfPlane(pixelBuffer, 0);
//    // 获取CMVImageBufferRef中的uv数据
//    UInt8 *pUV = (UInt8 *)CVPixelBufferGetBaseAddressOfPlane(pixelBuffer, 1);
//
//    size_t width = CVPixelBufferGetWidth(pixelBuffer);
//    size_t height = CVPixelBufferGetHeight(pixelBuffer);
//    size_t pYBytes = CVPixelBufferGetBytesPerRowOfPlane(pixelBuffer, 0);
//    size_t pUVBytes = CVPixelBufferGetBytesPerRowOfPlane(pixelBuffer, 1);
    size_t width = 1440;
    size_t height = 1920;
    
    
    UInt8 *pYUV420P = (UInt8 *)malloc(width * height * 3 / 2);
    memset(pYUV420P,117,width * height * 3 / 2);
    UInt8 *pU = pYUV420P + (width * height);
    UInt8 *pV = pU + (width * height / 4);
    
//    for(int i = 0; i < height; i++) {
//        memcpy(pYUV420P + i * width, pY + i * pYBytes, width);
//    }
//
//    for(int j = 0; j < height / 2; j++) {
//        for(int i = 0; i < width / 2; i++) {
//            *(pU++) = pUV[i<<1];
//            *(pV++) = pUV[(i<<1) + 1];
//        }
//
//        pUV += pUVBytes;
//    }
    
//    CVPixelBufferRef getCroppedPixelBuffer = [self copyDataFromBuffer:pYUV420P toYUVPixelBufferWithWidth:width Height:height];

    
//    NSDate *currentDate = [NSDate date];//获取当前时间，日期
//    NSDateFormatter *dateFormatter = [[NSDateFormatter alloc] init];// 创建一个时间格式化对象
//    [dateFormatter setDateFormat:@"YYYY-MM-dd hh:mm:ss SS"];//设定时间格式,这里可以设置成自己需要的格式
//    NSString *dateString = [dateFormatter stringFromDate:currentDate];//将时间转化成字符串
//    wchar_t* res = (wchar_t*)[dateString cStringUsingEncoding:NSUTF32StringEncoding];
    
    // 加水印
//    txtOverlay(self.txt, pYUV420P, res, [dateString length], 10, 40);
    if (!pCodecCtx) {
        setupEncoderWithWidth(width, height);
    }
    pFrame->data[0] = pYUV420P;
    pFrame->data[1] = pFrame->data[0] + width * height;
    pFrame->data[2] = pFrame->data[1] + (width * height) / 4;
    pFrame->pts = frameCounter;
    

    {
        
//        NSLog(@"%lu*%lu",width,height);
//        NSString *path = [self GetFilePathByfileName:[NSString stringWithFormat:@"YUV420P22_%lux%lu.YUV",width,height]];
//        NSLog(@"%@", path);
        
        FILE* file =fopen("YUV420P33.YUV", "wb");
        
//            fwrite(data , 1, size, file);
        
        int y_size = width * height;


        //Encode
        // 给AVPacket分配足够大的空间
        AVPacket pkt;
        av_new_packet(&pkt, y_size * 3);

        int size = av_image_get_buffer_size((AVPixelFormat)(pFrame->format), width, height, 1);

        int ret = 0;
        if ((ret = av_image_copy_to_buffer(pkt.data, pkt.size,
                                           pFrame->data, pFrame->linesize,
                                           (AVPixelFormat)(pFrame->format),
                                           pFrame->width, pFrame->height, 1)) < 0)
            return ;
        
        fwrite(pkt.data , 1, size, file);
        
        fclose(file);
    }
    
    
    int got_picture = 0;
    
    if (!pCodecCtx) {
//        CVPixelBufferUnlockBaseAddress(pixelBuffer, 0);
        return;
    }
    
    int ret = avcodec_encode_video2(pCodecCtx, &packet, pFrame, &got_picture);
    
    if(ret < 0) {
//        NSLog(@"Failed to encode!");
    }
    
    if (got_picture == 1) {
        fwrite(packet.data, 1, packet.size, fp);
        
//        if (self.delegate != nil) {
//            CMTime timestamp = CMSampleBufferGetPresentationTimeStamp(sample);
//
//            CFDictionaryRef ref = (CFDictionaryRef)CFArrayGetValueAtIndex(CMSampleBufferGetSampleAttachmentsArray(sample, true), 0);
            // 判断当前帧是否为关键帧
//            bool keyframe = !CFDictionaryContainsKey(ref , kCMSampleAttachmentKey_NotSync);
            
//            NSData *data = [NSData dataWithBytes:packet.data length:packet.size];
            
//            [self parseH264Data:data keyframe:keyframe];
            
//            if (self->enabledWriteVideoFile) {
//
//                fwrite(data.bytes, 1, data.length, self->fp);
//            }
//            [self.delegate gotX264EncoderData:data keyFrame:keyframe timestamp:timestamp error:nil];
//        }
        
//        NSLog(@"Succeed to encode frame: %5d\tsize:%5d", frameCounter, packet.size);
        frameCounter++;
        av_free_packet(&packet);
    }
    
    free(pYUV420P);
//    CVPixelBufferUnlockBaseAddress(pixelBuffer, 0);
}

int ccmain(int argc, const char * argv[]) {
    // insert code here...
    
//    encoding();
    
    return 0;
}

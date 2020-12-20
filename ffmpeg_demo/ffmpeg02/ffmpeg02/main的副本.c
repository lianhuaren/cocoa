//
//  main.c
//  ffmpeg02
//
//  Created by libb on 2020/12/18.
//
#include <libavformat/avformat.h>

#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libswscale/swscale.h>
#include <libavfilter/avfilter.h>
#include <stdio.h>

#define MAX_PATH 128

/*
 * Video decoding example
 */

static void pgm_save(unsigned char *buf, int wrap, int xsize, int ysize,
                     char *filename)
{
    FILE *f;
    int i;

    printf("pgm_save wrap:%d, xsize:%d, ysize:%d\n",wrap,xsize,ysize);
    f = fopen(filename,"w");
    fprintf(f, "P5\n%d %d\n%d\n", xsize, ysize, 255);
    for (i = 0; i < ysize; i++)
        fwrite(buf + i * wrap, 1, xsize, f);
    fclose(f);
}

/**
 * 将AVFrame(YUV420格式)保存为JPEG格式的图片
 *
 * @param width YUV420的宽
 * @param height YUV42的高
 *
 */
static int MyWriteJPEG(AVFrame* pFrame, int width, int height, int iIndex)
{
    if (iIndex%25 != 0)
    {
        return 0;
    }

    FILE* file =fopen("/Users/yanjun/Desktop/lbb/day/webrtcgit/input/tupian/YUV420P.YUV", "w");
    
    int i;
    for (i = 0; i<height; i++)
    {
        fwrite(pFrame->data[0]+i*pFrame->linesize[0] , 1, width, file);
    }
    for (i = 0; i<height / 2; i++)
    {
        fwrite(pFrame->data[1]+i*pFrame->linesize[1] , 1, width/2, file);

    }
    for (i = 0; i<height / 2; i++)
    {
        fwrite(pFrame->data[2]+i*pFrame->linesize[2] , 1, width/2, file);
    }
    
    fclose(file);
    
    int num = iIndex/25;
    // 输出文件路径
    char out_file[MAX_PATH] = {0};
     
     sprintf(out_file,"/Users/yanjun/Desktop/lbb/day/webrtcgit/input/tupian/%d.pgm",iIndex);
     
     pgm_save(pFrame->data[0], pFrame->linesize[0],
                      pFrame->width, pFrame->height, out_file);
    
    return 0;
      //sprintf_s(out_file, sizeof(out_file), "./%d.jpg",  iIndex);
      sprintf(out_file, "/Users/yanjun/Desktop/lbb/day/webrtcgit/input/tupian/%d.jpg",  num);
     
    // 分配AVFormatContext对象
    AVFormatContext* pFormatCtx = avformat_alloc_context();//avformat_alloc_context();
     
    // 设置输出文件格式
    pFormatCtx->oformat = av_guess_format("mjpeg", NULL, NULL);
     
    // 创建并初始化一个和该url相关的AVIOContext
    if( avio_open(&pFormatCtx->pb, out_file, AVIO_FLAG_READ_WRITE) < 0)
    {
    printf("Couldn't open output file.");
    return -1;
    }
     
     
    // 构建一个新stream
    AVStream* pAVStream = avformat_new_stream(pFormatCtx, 0);
    if( pAVStream == NULL )
    {
    return -1;
    }
     
     
    // 设置该stream的信息
    AVCodecContext* pCodecCtx = pAVStream->codec;
     
    pCodecCtx->codec_id = pFormatCtx->oformat->video_codec;
    pCodecCtx->codec_type = AVMEDIA_TYPE_VIDEO;
    pCodecCtx->pix_fmt = AV_PIX_FMT_YUVJ420P;
    pCodecCtx->width = width;
    pCodecCtx->height = height;
    pCodecCtx->time_base.num = 1;
    pCodecCtx->time_base.den = 25;
     
    // Begin Output some information
    av_dump_format(pFormatCtx, 0, out_file, 1);
    // End Output some information
     
     
    // 查找解码器
    AVCodec* pCodec = avcodec_find_encoder(pCodecCtx->codec_id);
    if( !pCodec )
    {
        printf("Codec not found.");
        return -1;
    }
    // 设置pCodecCtx的解码器为pCodec
    if( avcodec_open2(pCodecCtx, pCodec, NULL) < 0 )
    {
        printf("Could not open codec.");
        return -1;
    }
     
     
    //Write Header
    avformat_write_header(pFormatCtx, NULL);
     
     
    int y_size = pCodecCtx->width * pCodecCtx->height;
     
     
    //Encode
    // 给AVPacket分配足够大的空间
    AVPacket pkt;
    av_new_packet(&pkt, y_size * 3);
     
     
    //
    int got_picture = 0;
    int ret = avcodec_encode_video2(pCodecCtx, &pkt, pFrame, &got_picture);
    if( ret < 0 )
    {
        printf("Encode Error.\n");
        return -1;
    }
    printf("got_picture %d \n",got_picture);
    if( got_picture == 1 )
    {
        //pkt.stream_index = pAVStream->index;
        ret = av_write_frame(pFormatCtx, &pkt);
    }
     
     
    av_free_packet(&pkt);
     
     
    //Write Trailer
    av_write_trailer(pFormatCtx);
     
     
    printf("Encode Successful.\n");
     
     
    if( pAVStream )
    {
    avcodec_close(pAVStream->codec);
    }
     
    avio_close(pFormatCtx->pb);
    avformat_free_context(pFormatCtx);
     
     
    return 0;
}
int aamain(int argc,char* argv[])
{
    int videoStream = -1;
    AVCodecContext *pCodecCtx;
    AVFormatContext *pFormatCtx;
    AVCodec *pCodec;
    AVFrame *pFrame, *pFrameRGB;
    struct SwsContext *pSwsCtx;
    const char *filename = "/Users/yanjun/Desktop/lbb/day/webrtcgit/input/aa.mp4";
    AVPacket packet;
    int frameFinished;
    int PictureSize;
    uint8_t *outBuff;
 
 
    //注册编解码器
    av_register_all();
 
    // 分配AVFormatContext
    pFormatCtx = avformat_alloc_context();
 
 
    //打开视频文件
    if( avformat_open_input(&pFormatCtx, filename, NULL, NULL) != 0 )
    {
        printf ("av open input file failed!\n");
        exit (1);
    }
     
     
    //获取流信息
    if( avformat_find_stream_info(pFormatCtx, NULL) < 0 )
    {
        printf ("av find stream info failed!\n");
        exit (1);
    }
     
    printf(" pFormatCtx->nb_streams %d \n", pFormatCtx->nb_streams);
     
     
        //获取视频流
    for( int i = 0; i < pFormatCtx->nb_streams; i++ )
    {
        if ( pFormatCtx->streams[i]->codec->codec_type == AVMEDIA_TYPE_VIDEO )
        {
            videoStream = i;
            break;
        }
    }
    if( videoStream == -1 )
    {
        printf ("find video stream failed!\n");
        exit (1);
    }
     
     
        // 寻找解码器
    pCodecCtx = pFormatCtx->streams[videoStream]->codec;
     
    pCodec = avcodec_find_decoder(pCodecCtx->codec_id);
     
    if( pCodec == NULL )
    {
        printf ("avcode find decoder failed!\n");
        exit (1);
    }
     
     
        //打开解码器
    if( avcodec_open2(pCodecCtx, pCodec, NULL) < 0 )
    {
        printf ("avcode open failed!\n");
        exit (1);
    }
     
     
        //为每帧图像分配内存
    pFrame = av_frame_alloc();//avcodec_alloc_frame();
     
     
    int i = 0;
     
    while( av_read_frame(pFormatCtx, &packet) >= 0 )
    {
        if( packet.stream_index == videoStream )
        {
            avcodec_decode_video2(pCodecCtx, pFrame, &frameFinished, &packet);
             
             
            if( frameFinished )
            {

                
                MyWriteJPEG(pFrame, pCodecCtx->width, pCodecCtx->height, i ++);
            }
        }
        else
        {
            int a=2;
            int b=a;
        }
         
         
        av_free_packet(&packet);
    }
     
     
//    sws_freeContext(pSwsCtx);
     
     
    av_free(pFrame);
    av_free(pFrameRGB);
    avcodec_close(pCodecCtx);
    avformat_close_input(&pFormatCtx);
 
 
    return 0;
}

void Java_com_ffmpeg_VideoUtils_decode()
{
    const char *input_cstr = "/Users/yanjun/Desktop/lbb/day/webrtcgit/input/aa.mp4";//(*env)->GetStringUTFChars(env, input_, 0);
    const char *output_cstr = "/Users/yanjun/Desktop/lbb/day/webrtcgit/input/";//(*env)->GetStringUTFChars(env, output_, 0);
    //    //需要转码的视频文件(输入的视频文件)

    //1.注册所有主键
//    av_register_all();
    //封装格式上下文，统领全局的结构体，保存了视频文件封装格式的相关信息
    AVFormatContext *avFormatContext = avformat_alloc_context();

    //2.打开输入视频文件夹
    int err_code = avformat_open_input(&avFormatContext, input_cstr, NULL, NULL);
    if (err_code != 0) {
        char errbuf[1024];
        const char *errbuf_ptr = errbuf;
        av_strerror(err_code, errbuf_ptr, sizeof(errbuf));
//        LOGE("Couldn't open file %s: %d(%s)", input_cstr, err_code, errbuf_ptr);
//        LOGE("%s", "打开输入视频文件失败");
        return;
    }

    //3.获取视频文件信息
    avformat_find_stream_info(avFormatContext, NULL);

    //获取视频流的索引位置
    //遍历所有类型的流（音频流、视频流、字幕流），找到视频流
    int v_stream_idx = -1;
    int i = 0;
    for (; i < avFormatContext->nb_streams; i++) {
        if (avFormatContext->streams[i]->codec->codec_type == AVMEDIA_TYPE_VIDEO) {
            v_stream_idx = i;
            break;
        }
    }

    if (v_stream_idx == -1) {
//        LOGE("%s", "找不到视频流\n");
        return;
    }

    //只有知道视频的编码方式，才能够根据编码方式去找到解码器
    //获取视频流中的编解码上下文
    AVCodecContext *pCodecCtx = avFormatContext->streams[v_stream_idx]->codec;
    //4.根据编解码上下文中的编码id查找对应的解码
    AVCodec *pCodec = avcodec_find_decoder(pCodecCtx->codec_id);
    if (pCodec == NULL) {
//        LOGE("%s", "找不到解码器\n");
        return;
    }


    //5.打开解码器
    if (avcodec_open2(pCodecCtx, pCodec, NULL) < 0) {
//        LOGE("%s", "解码器无法打开\n");
        return;
    };

    //准备读取
    //AVPacket用于存储一帧一帧的压缩数据（H264）
    //缓冲区，开辟空间
    AVPacket *packet = (AVPacket *) av_malloc(sizeof(AVPacket));

    //AVFrame用于存储解码后的像素数据(YUV)
    //内存分配
    AVFrame *pFrame = av_frame_alloc();
    //YUV420
    AVFrame *pFrameYUV = av_frame_alloc();

    //只有指定了AVFrame的像素格式、画面大小才能真正分配内存
    //缓冲区分配内存
    uint8_t *out_buffer = (uint8_t *) av_malloc(
            avpicture_get_size(AV_PIX_FMT_YUV420P, pCodecCtx->width, pCodecCtx->height));
    //初始化缓冲区
    avpicture_fill((AVPicture *) pFrameYUV, out_buffer, AV_PIX_FMT_YUV420P, pCodecCtx->width,
                   pCodecCtx->height);


//    //用于转码（缩放）的参数，转之前的宽高，转之后的宽高，格式等
//    struct SwsContext *sws_ctx = sws_getContext(pCodecCtx->width, pCodecCtx->height,
//                                                pCodecCtx->pix_fmt,
//                                                pCodecCtx->width, pCodecCtx->height,
//                                                AV_PIX_FMT_YUV420P,
//                                                SWS_BICUBIC, NULL, NULL, NULL);


    int got_picture, ret;

    FILE *fp_yuv = fopen(output_cstr, "wb+");

    int frame_count = 0;

//    6.一帧一帧的读取压缩数据
    int readCode = av_read_frame(avFormatContext, packet);
//    LOGI("av_read_frame error = %d", readCode);
    while ( readCode>= 0) {

        //只要视频压缩数据（根据流的索引位置判断）
        if (packet->stream_index == v_stream_idx) {
            //7.解码一帧视频压缩数据，得到视频像素数据
            ret = avcodec_decode_video2(pCodecCtx, pFrame, &got_picture, packet);
            if (ret < 0) {
//                LOGE("%s", "解码错误");
                return;
            }

            //为0说明解码完成，非0正在解码
            if (got_picture) {
                //AVFrame转为像素格式YUV420，宽高
                //2 6输入、输出数据
                //3 7输入、输出画面一行的数据的大小 AVFrame 转换是一行一行转换的
                //4 输入数据第一列要转码的位置 从0开始
                //5 输入画面的高度
//                sws_scale(sws_ctx, (const uint8_t *const *) pFrame->data, pFrame->linesize, 0, pCodecCtx->height,
//                          pFrameYUV->data, pFrameYUV->linesize);

                //输出到YUV文件
                //AVFrame像素帧写入文件
                //data解码后的图像像素数据（音频采样数据）
                //Y 亮度 UV 色度（压缩了） 人对亮度更加敏感
                //U V 个数是Y的1/4
                int y_size = pCodecCtx->width * pCodecCtx->height;
                fwrite(pFrameYUV->data[0], 1, y_size, fp_yuv);
                fwrite(pFrameYUV->data[1], 1, y_size / 4, fp_yuv);
                fwrite(pFrameYUV->data[2], 1, y_size / 4, fp_yuv);

                frame_count++;
//                LOGI("解码第%d帧", frame_count);
            }
        }

        //释放资源
        av_free_packet(packet);
        readCode = av_read_frame(avFormatContext, packet);
//        LOGI("av_read_frame error = %d", readCode);
    }

    fclose(fp_yuv);

//    (*env)->ReleaseStringUTFChars(env, input_, input_cstr);
//    (*env)->ReleaseStringUTFChars(env, output_, output_cstr);

    av_frame_free(&pFrame);

    avcodec_close(pCodecCtx);

    avformat_free_context(avFormatContext);
}

int main(int argc, const char * argv[]) {
    // insert code here...
//    avcodec_register_all();
//    Java_com_ffmpeg_VideoUtils_decode();
    aamain(argc,argv);
    
    return 0;
}

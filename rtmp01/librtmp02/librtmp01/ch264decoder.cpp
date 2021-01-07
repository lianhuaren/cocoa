#include "ch264decoder.h"

#define FF_INPUT_BUFFER_PADDING_SIZE 16

typedef unsigned char  BYTE;

typedef unsigned short WORD;

typedef unsigned long  DWORD;

//#include <QDebug>

CH264Decoder::CH264Decoder()
{
    createYUVTab_16();
}

CH264Decoder::~CH264Decoder()
{
    deleteYUVTab();
}

void CH264Decoder::deleteYUVTab()
{
    av_free(colortab);
    av_free(rgb_2_pix);
}

void CH264Decoder::createYUVTab_16()
{
    int i;
    int u, v;

    colortab = (int *)av_malloc(4*256*sizeof(int));
    u_b_tab = &colortab[0*256];
    u_g_tab = &colortab[1*256];
    v_g_tab = &colortab[2*256];
    v_r_tab = &colortab[3*256];

    for (i=0; i<256; i++)
    {
        u = v = (i-128);

        u_b_tab[i] = (int) ( 1.772 * u);
        u_g_tab[i] = (int) ( 0.34414 * u);
        v_g_tab[i] = (int) ( 0.71414 * v);
        v_r_tab[i] = (int) ( 1.402 * v);
    }

    rgb_2_pix = (unsigned int *)av_malloc(3*768*sizeof(unsigned int));

    r_2_pix = &rgb_2_pix[0*768];
    g_2_pix = &rgb_2_pix[1*768];
    b_2_pix = &rgb_2_pix[2*768];

    for(i=0; i<256; i++)
    {
        r_2_pix[i] = 0;
        g_2_pix[i] = 0;
        b_2_pix[i] = 0;
    }

    for(i=0; i<256; i++)
    {
        r_2_pix[i+256] = (i & 0xF8) << 8;
        g_2_pix[i+256] = (i & 0xFC) << 3;
        b_2_pix[i+256] = (i ) >> 3;
    }

    for(i=0; i<256; i++)
    {
        r_2_pix[i+512] = 0xF8 << 8;
        g_2_pix[i+512] = 0xFC << 3;
        b_2_pix[i+512] = 0x1F;
    }

    r_2_pix += 256;
    g_2_pix += 256;
    b_2_pix += 256;
}

void CH264Decoder::displayYUV_16(unsigned int *pdst, unsigned char *y, unsigned char *u, unsigned char *v, int width, int height, int src_ystride, int src_uvstride, int dst_ystride)
{
    int i, j;
    int r, g, b, rgb;

    int yy, ub, ug, vg, vr;

    unsigned char* yoff;
    unsigned char* uoff;
    unsigned char* voff;

    int width2 = width/2;
    int height2 = height/2;

    for(j=0; j<height2; j++)
    {
        yoff = y + j * 2 * src_ystride;
        uoff = u + j * src_uvstride;
        voff = v + j * src_uvstride;

        for(i=0; i<width2; i++)
        {
            yy  = *(yoff+(i<<1));
            ub = u_b_tab[*(uoff+i)];
            ug = u_g_tab[*(uoff+i)];
            vg = v_g_tab[*(voff+i)];
            vr = v_r_tab[*(voff+i)];

            b = yy + ub;
            g = yy - ug - vg;
            r = yy + vr;

            rgb = r_2_pix[r] + g_2_pix[g] + b_2_pix[b];

            yy = *(yoff+(i<<1)+1);
            b = yy + ub;
            g = yy - ug - vg;
            r = yy + vr;

            pdst[(j*dst_ystride+i)] = (rgb)+((r_2_pix[r] + g_2_pix[g] + b_2_pix[b])<<16);

            yy = *(yoff+(i<<1)+src_ystride);
            b = yy + ub;
            g = yy - ug - vg;
            r = yy + vr;

            rgb = r_2_pix[r] + g_2_pix[g] + b_2_pix[b];

            yy = *(yoff+(i<<1)+src_ystride+1);
            b = yy + ub;
            g = yy - ug - vg;
            r = yy + vr;

            pdst [((2*j+1)*dst_ystride+i*2)>>1] = (rgb)+((r_2_pix[r] + g_2_pix[g] + b_2_pix[b])<<16);
        }
    }
}
int CH264Decoder::initial()
{



    avcodec_register_all();
    av_init_packet(&packet);

    codec = avcodec_find_decoder(AV_CODEC_ID_H264);
    if (!codec)
    {
        printf("avcodec_find_encoder failed");
        return -1;
    }

    context = avcodec_alloc_context3(codec);
    if (!context)
    {
        printf("avcodec_alloc_context3 failed");
        return -2;
    }

    context->codec_type = AVMEDIA_TYPE_VIDEO;
    context->pix_fmt = AV_PIX_FMT_YUV420P;

    if (avcodec_open2(context, codec, NULL) < 0)
    {
        printf("avcodec_open2 failed");
        return -3;
    }

    frame = av_frame_alloc();
    if (!frame)
    {
        return -4;
    }

    return 0;
}

int CH264Decoder::initial2()
{
    const char *fname = "/Users/yanjun/Desktop/lbb/day/ffmpeg/test.h264";
    
    av_register_all();
    avformat_network_init();
    AVFormatContext *pFormatContext = avformat_alloc_context();
    if (avformat_open_input(&pFormatContext, fname, NULL, NULL) != 0) {
        printf("Couldn't open input stream.\n");
        return -1;
    }
    if (avformat_find_stream_info(pFormatContext, NULL) < 0) {
        printf("Couldn't find stream information.\n");
        return -1;
    }
    int videoindex = -1;
    for (int i=0; i<pFormatContext->nb_streams; i++) {
        if (pFormatContext->streams[i]->codec->codec_type == AVMEDIA_TYPE_VIDEO) {
            videoindex = i;
            break;
        }
    }
    if (videoindex==-1) {
        printf("Didn't find a video stream.\n");
        return -1;
    }
    
    AVCodecContext *pCodecCtx = pFormatContext->streams[videoindex]->codec;
    AVCodec *pCodec = avcodec_find_decoder(pCodecCtx->codec_id);
    if (pCodec == NULL) {
        printf("Codec not found.\n");
        return -1;
    }
    if (avcodec_open2(pCodecCtx, pCodec, NULL)<0) {
        printf("Could not open codec.\n");
        return -1;
    }
    
    AVFrame *pFrame = av_frame_alloc();
    AVFrame *pFrameYUV = av_frame_alloc();
    int buffer_size = av_image_get_buffer_size(AV_PIX_FMT_YUV420P, pCodecCtx->width, pCodecCtx->height, 1);
    unsigned char *out_buffer = (unsigned char *)av_malloc(buffer_size);
    
    AVPacket *packet = (AVPacket *)av_malloc(sizeof(AVPacket));
    
    
    
    // 打开h264文件，并把文件信息存入fctx中
    int iRes = 0;
    AVFormatContext *fctx = avformat_alloc_context();
    if ((iRes = avformat_open_input(&fctx, fname, NULL, NULL)) != 0)
    {
        cout << "File open failed!" << endl;
        return -1;
    }
    // 寻找视频流信息
    if (avformat_find_stream_info(fctx, NULL) < 0)
    {
        cout << "Stream find failed!\n";
        return -1;
    }
    // dump调试信息
    av_dump_format(fctx, -1, fname, NULL);
    // 打开了视频并且获取了视频流 ，设置视频索引值默认值
    int vindex = -1;
    for (int i = 0; i < fctx->nb_streams; i++)
    {
//        if (fctx->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_VIDEO)
        if(fctx->streams[i]->codec->codec_type==AVMEDIA_TYPE_VIDEO)
            vindex = i;
    }
    // 如果没有找到视频的索引，说明并不是一个视频文件
    if (vindex == -1)
    {
        cout << "Codec find failed!" << endl;
        return -1;
    }
//    // 分配解码器上下文空间
//    AVCodecContext *cctx = avcodec_alloc_context3(NULL);
//    // 获取编解码器上下文信息
//    if (avcodec_parameters_to_context(cctx, fctx->streams[vindex]->codecpar) < 0)
//    {
//        cout << "Copy stream failed!" << endl;
//        return -1;
//    }
    AVCodecContext *cctx = fctx->streams[vindex]->codec;
    // 查找解码器
    AVCodec *c = avcodec_find_decoder(cctx->codec_id);
    if (!c) {
        cout << "Find Decoder failed!" << endl;
        return -1;
    }
    // 打开解码器
    if (avcodec_open2(cctx, c, NULL) != 0) {
        cout << "Open codec failed!" << endl;
        return -1;
    }
    // 对图形进行宽度上方的裁剪，以便于显示得更好
    struct SwsContext *imgCtx = sws_getContext(cctx->width, cctx->height, cctx->pix_fmt, cctx->width, cctx->height, AV_PIX_FMT_YUV420P, SWS_BICUBIC, NULL, NULL, NULL);
    if (!imgCtx)
    {
        cout << "Get swscale context failed!" << endl;
        return -1;
    }
    AVPacket *pkt = (AVPacket *)av_malloc(sizeof(AVPacket));//av_packet_alloc();
    AVFrame *fr = av_frame_alloc();
    AVFrame *yuv = av_frame_alloc();
    int vsize = av_image_get_buffer_size(AV_PIX_FMT_YUV420P, cctx->width, cctx->height, 1);
    uint8_t *buf = (uint8_t *)av_malloc(vsize);
    av_image_fill_arrays(yuv->data, yuv->linesize, buf, AV_PIX_FMT_YUV420P, cctx->width, cctx->height, 1);
    char errbuf[256] = { 0 };
    FILE *fp_yuv = fopen("/Users/yanjun/Desktop/lbb/day/ffmpeg/test_out.yuv", "wb+");
//    while (av_read_frame(fctx, pkt) >= 0)
//    {
//        if (pkt->stream_index == vindex)
//        {
//            if ((iRes = avcodec_send_packet(cctx, pkt)) != 0)
//            {
//                cout << "Send video stream packet failed!" << endl;
//                av_strerror(iRes, errbuf, 256);
//                return -5;
//            }
//            if ((iRes = avcodec_receive_frame(cctx, fr)) != 0)
//            {
//                cout << "Receive video frame failed!" << endl;
//                av_strerror(iRes, errbuf, 256);
//                return -6;
//            }
//            cout << "decoding the frame " << cctx->frame_number << endl;
//            sws_scale(imgCtx, fr->data, fr->linesize, 0, cctx->height, yuv->data, yuv->linesize);
//            int y_size = cctx->width*cctx->height;
//            fwrite(yuv->data[0], 1, y_size, fp_yuv);        // Y
//            fwrite(yuv->data[1], 1, y_size / 4, fp_yuv);    // U
//            fwrite(yuv->data[2], 1, y_size / 4, fp_yuv);    // V
//            
//        }
//    }
    int got_picture = 0;
    int ret = 0;
    int y_size;
    
    while(av_read_frame(fctx, pkt)>=0){
        if(pkt->stream_index==vindex){
            ret = avcodec_decode_video2(cctx, fr, &got_picture, pkt);
            if(ret < 0){
                printf("Decode Error.\n");
                return -1;
            }
            if(got_picture){
                sws_scale(imgCtx, (const unsigned char* const*)fr->data, fr->linesize, 0, cctx->height,
                          yuv->data, yuv->linesize);
                
                y_size=cctx->width*cctx->height;
                fwrite(yuv->data[0],1,y_size,fp_yuv);    //Y
                fwrite(yuv->data[1],1,y_size/4,fp_yuv);  //U
                fwrite(yuv->data[2],1,y_size/4,fp_yuv);  //V
                printf("Succeed to decode 1 frame!\n");
                
            }
        }
        av_free_packet(pkt);
    }
    av_free(buf);
    av_frame_free(&yuv);
    av_frame_free(&fr);
//    av_packet_free(&pkt);
    sws_freeContext(imgCtx);
    avcodec_free_context(&cctx);
    avformat_close_input(&fctx);
    avformat_free_context(fctx);
    
    return 0;
}

#pragma pack(push,1)
typedef struct tagBITMAPFILEHEADER {
    uint16_t    bfType;
    uint32_t    bfSize;
    uint32_t    bfReserved;
    uint32_t    bfOffBits;
} BITMAPFILEHEADER;

typedef struct tagBITMAPINFOHEADER {
    uint32_t    biSize;
    int32_t     biWidth;
    int32_t     biHeight;
    uint16_t    biPlanes;
    uint16_t    biBitCount;
    uint32_t    biCompression;
    uint32_t    biSizeImage;
    int32_t     biXPelsPerMeter;
    int32_t     biYPelsPerMeter;
    uint32_t    biClrUsed;
    uint32_t    biClrImportant;
} BITMAPINFOHEADER;
#pragma pack(pop)

typedef struct tagRGBQUAD{
    BYTE    rgbBlue;
    BYTE rgbGreen;
    BYTE rgbRed;
    BYTE rgbReserved;
}RGBQUAD;

typedef   struct   tagBITMAPINFO   {
    BITMAPINFOHEADER         bmiHeader;
    RGBQUAD                           bmiColors[1];
}   BITMAPINFO;



static int av_create_bmp(char* filename,uint8_t *pRGBBuffer,int width,int height,int bpp)
{
    BITMAPFILEHEADER bmpheader;
    BITMAPINFO bmpinfo;
    FILE *fp;
    
    fp = fopen(filename,"wb");
    if(!fp)return -1;
    
    bmpheader.bfType = ('M'<<8)|'B';
    bmpheader.bfReserved = 0;
//    bmpheader.bfReserved2 = 0;
    bmpheader.bfOffBits = sizeof(BITMAPFILEHEADER) + sizeof(BITMAPINFOHEADER);
    bmpheader.bfSize = bmpheader.bfOffBits + width*height*bpp/8;
    
    bmpinfo.bmiHeader.biSize = sizeof(BITMAPINFOHEADER);
    bmpinfo.bmiHeader.biWidth = width;
    bmpinfo.bmiHeader.biHeight = height;
    bmpinfo.bmiHeader.biPlanes = 1;
    bmpinfo.bmiHeader.biBitCount = bpp;
    bmpinfo.bmiHeader.biCompression = 0;
    bmpinfo.bmiHeader.biSizeImage = 0;
    bmpinfo.bmiHeader.biXPelsPerMeter = 100;
    bmpinfo.bmiHeader.biYPelsPerMeter = 100;
    bmpinfo.bmiHeader.biClrUsed = 0;
    bmpinfo.bmiHeader.biClrImportant = 0;
    
    fwrite(&bmpheader,sizeof(BITMAPFILEHEADER),1,fp);
    fwrite(&bmpinfo.bmiHeader,sizeof(BITMAPINFOHEADER),1,fp);
    fwrite(pRGBBuffer,width*height*bpp/8,1,fp);
    fclose(fp);
    
    return 0;
}

int CH264Decoder::decode3(uint8_t *pDataIn, int nInSize, uint8_t *pDataOut, int *nWidth, int *nHeight)
{
    
    
    av_init_packet(&packet);
    packet.size = nInSize;
    packet.data = pDataIn;
    
    
    uint8_t *yuvBuffer;
    //    AVFrame *pFrame ;
    AVFrame *pFrameRGB;
    uint8_t * rgbBuffer;
    SwsContext *img_convert_ctx;
    
    static int i=0;
    if (packet.size > 0)
    {
        int got_picture=0;
        int ret= avcodec_decode_video2(context, frame, &got_picture, &packet);
        if (ret < 0)
        {
            printf("avcodec_encode_video2 failed");
            return -2;
        }
        
        if (got_picture)
        {
            
            //             displayYUV_16((unsigned int*)pDataOut, frame->data[0], frame->data[1],frame->data[2],
            //                     *nWidth,*nHeight,frame->linesize[0],frame->linesize[2],*nWidth);
            
            int width = context->width;
            int height = context->height;
            
            //width和heigt为传入的分辨率的大小，实际应用我传的1280*720
            int yuvSize = width * height * 3 /2;
            yuvBuffer = (uint8_t *)malloc(yuvSize);
            //为每帧图像分配内存
            //            pFrame = av_frame_alloc();
            pFrameRGB = av_frame_alloc();
            int numBytes = avpicture_get_size(AV_PIX_FMT_RGB32, width,height);
            rgbBuffer = (uint8_t *) av_malloc(numBytes * sizeof(uint8_t));
            avpicture_fill((AVPicture *) pFrameRGB, rgbBuffer, AV_PIX_FMT_RGB32,width, height);
            //特别注意 img_convert_ctx 该在做H264流媒体解码时候，发现sws_getContext /sws_scale内存泄露问题，
            //注意sws_getContext只能调用一次，在初始化时候调用即可，另外调用完后，在析构函数中使用sws_free_Context，将它的内存释放。
            //设置图像转换上下文
            img_convert_ctx = sws_getContext(width, height, AV_PIX_FMT_YUV420P, width, height, AV_PIX_FMT_RGB32, SWS_BICUBIC, NULL, NULL, NULL);
            
            //            avpicture_fill((AVPicture *) pFrame, (uint8_t *)str, AV_PIX_FMT_YUV420P, width, height);//这里的长度和高度跟之前保持一致
            //转换图像格式，将解压出来的YUV420P的图像转换为RGB的图像
            sws_scale(img_convert_ctx,
                      (uint8_t const * const *) frame->data,
                      frame->linesize, 0, height, pFrameRGB->data,
                      pFrameRGB->linesize);

            char pic[200];
            sprintf(pic,"/Users/yanjun/Desktop/lbb/day/ffmpeg/bmp/image%d.bmp",i);
            i++;
            
            if (i<50) {
                av_create_bmp(pic,rgbBuffer,width,height,32);
            }
            
            //把这个RGB数据 用QImage加载
//            QImage tmpImg((uchar *)rgbBuffer,width,height,QImage::Format_RGB32);
//            //            QImage image = tmpImg.copy(); //把图像复制一份 传递给界面显示
//
//            tmpImg.save("/Users/yanjun/Desktop/lbb/day/ffmpeg/1.jpg", "JPG", 100);
        }
    }
    else
    {
        printf("no data to decode");
        return -1;
    }
    
    
    return 0;
}

void CH264Decoder::unInitial()
{
    avcodec_close(context);
    av_free(context);
    av_frame_free(&frame);
}

int CH264Decoder::test()
{
    AVCodec *pCodec;
    AVCodecContext *pCodecCtx= NULL;
    AVCodecParserContext *pCodecParserCtx=NULL;
    
    FILE *fp_in;
    FILE *fp_out;
    AVFrame    *pFrame;
    
    const int in_buffer_size=4096;
    unsigned char in_buffer[in_buffer_size + FF_INPUT_BUFFER_PADDING_SIZE]={0};
    unsigned char *cur_ptr;
    int cur_size;
    AVPacket packet;
    int ret, got_picture;
    
    

    AVCodecID codec_id=AV_CODEC_ID_H264;
    char filepath_in[]="/Users/yanjun/Desktop/lbb/day/ffmpeg/bigbuckbunny_480x272.h264";

    
    char filepath_out[]="/Users/yanjun/Desktop/lbb/day/bigbuckbunny_480x272.yuv";
    int first_time=1;
    
    
    //av_log_set_level(AV_LOG_DEBUG);
    
    avcodec_register_all();
    
    pCodec = avcodec_find_decoder(codec_id);
    if (!pCodec) {
        printf("Codec not found\n");
        return -1;
    }
    pCodecCtx = avcodec_alloc_context3(pCodec);
    if (!pCodecCtx){
        printf("Could not allocate video codec context\n");
        return -1;
    }
    
    pCodecParserCtx=av_parser_init(codec_id);
    if (!pCodecParserCtx){
        printf("Could not allocate video parser context\n");
        return -1;
    }
    
    //if(pCodec->capabilities&CODEC_CAP_TRUNCATED)
    //    pCodecCtx->flags|= CODEC_FLAG_TRUNCATED;
    
    if (avcodec_open2(pCodecCtx, pCodec, NULL) < 0) {
        printf("Could not open codec\n");
        return -1;
    }
    //Input File
    fp_in = fopen(filepath_in, "rb");
    if (!fp_in) {
        printf("Could not open input stream\n");
        return -1;
    }
    //Output File
    fp_out = fopen(filepath_out, "wb");
    if (!fp_out) {
        printf("Could not open output YUV file\n");
        return -1;
    }
    
    pFrame = av_frame_alloc();
    av_init_packet(&packet);
    
    while (1) {
        
        cur_size = fread(in_buffer, 1, in_buffer_size, fp_in);
        if (cur_size == 0)
            break;
        cur_ptr=in_buffer;
        
        while (cur_size>0){
            
            int len = av_parser_parse2(
                                       pCodecParserCtx, pCodecCtx,
                                       &packet.data, &packet.size,
                                       cur_ptr , cur_size ,
                                       AV_NOPTS_VALUE, AV_NOPTS_VALUE, AV_NOPTS_VALUE);
            
            cur_ptr += len;
            cur_size -= len;
            
            if(packet.size==0)
                continue;
            
            //Some Info from AVCodecParserContext
            printf("[Packet]Size:%6d\t",packet.size);
            switch(pCodecParserCtx->pict_type){
                case AV_PICTURE_TYPE_I: printf("Type:I\t");break;
                case AV_PICTURE_TYPE_P: printf("Type:P\t");break;
                case AV_PICTURE_TYPE_B: printf("Type:B\t");break;
                default: printf("Type:Other\t");break;
            }
            printf("Number:%4d\n",pCodecParserCtx->output_picture_number);
            
            ret = avcodec_decode_video2(pCodecCtx, pFrame, &got_picture, &packet);
            if (ret < 0) {
                printf("Decode Error.\n");
                return ret;
            }
            if (got_picture) {
                if(first_time){
                    printf("\nCodec Full Name:%s\n",pCodecCtx->codec->long_name);
                    printf("width:%d\nheight:%d\n\n",pCodecCtx->width,pCodecCtx->height);
                    first_time=0;
                }
                //Y, U, V
                for(int i=0;i<pFrame->height;i++){
                    fwrite(pFrame->data[0]+pFrame->linesize[0]*i,1,pFrame->width,fp_out);
                }
                for(int i=0;i<pFrame->height/2;i++){
                    fwrite(pFrame->data[1]+pFrame->linesize[1]*i,1,pFrame->width/2,fp_out);
                }
                for(int i=0;i<pFrame->height/2;i++){
                    fwrite(pFrame->data[2]+pFrame->linesize[2]*i,1,pFrame->width/2,fp_out);
                }
                
                printf("Succeed to decode 1 frame!\n");
            }
        }
        
    }
    
    //Flush Decoder
    packet.data = NULL;
    packet.size = 0;
    while(1){
        ret = avcodec_decode_video2(pCodecCtx, pFrame, &got_picture, &packet);
        if (ret < 0) {
            printf("Decode Error.\n");
            return ret;
        }
        if (!got_picture){
            break;
        }else {
            //Y, U, V
            for(int i=0;i<pFrame->height;i++){
                fwrite(pFrame->data[0]+pFrame->linesize[0]*i,1,pFrame->width,fp_out);
            }
            for(int i=0;i<pFrame->height/2;i++){
                fwrite(pFrame->data[1]+pFrame->linesize[1]*i,1,pFrame->width/2,fp_out);
            }
            for(int i=0;i<pFrame->height/2;i++){
                fwrite(pFrame->data[2]+pFrame->linesize[2]*i,1,pFrame->width/2,fp_out);
            }
            
            printf("Flush Decoder: Succeed to decode 1 frame!\n");
        }
    }
    
    fclose(fp_in);
    fclose(fp_out);
    
    
    av_parser_close(pCodecParserCtx);
    
    av_frame_free(&pFrame);
    avcodec_close(pCodecCtx);
    av_free(pCodecCtx);
    
    return 0;
}

int CH264Decoder::decode(uint8_t *pDataIn, int nInSize, uint8_t *pDataOut,int *nWidth, int *nHeight)
{
    av_init_packet(&packet);
    packet.size = nInSize;
    packet.data = pDataIn;

    if (packet.size > 0)
    {
        int got_picture=0;
        int ret= avcodec_decode_video2(context, frame, &got_picture, &packet);
        if (ret < 0)
        {
            printf("avcodec_encode_video2 failed");
            return -2;
        }

        if (got_picture)
        {
            *nWidth = context->width;
            *nHeight = context->height;

             displayYUV_16((unsigned int*)pDataOut, frame->data[0], frame->data[1],frame->data[2],
                     *nWidth,*nHeight,frame->linesize[0],frame->linesize[2],*nWidth);
        }
    }
    else
    {
        printf("no data to decode");
        return -1;
    }

    return 0;
}

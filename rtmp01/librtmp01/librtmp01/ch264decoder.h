#ifndef CH264DECODER_H
#define CH264DECODER_H

#include <string.h>
#include <iostream>

//C++引用C语言的头文件
extern "C"
{
#include "libavformat/avformat.h"
#include "libswresample/swresample.h"
#include "libswscale/swscale.h"
#include "libavutil/opt.h"
#include "libavutil/channel_layout.h"
#include "libavutil/parseutils.h"
#include "libavutil/samplefmt.h"
#include "libavutil/fifo.h"
#include "libavutil/intreadwrite.h"
#include "libavutil/dict.h"
#include "libavutil/mathematics.h"
#include "libavutil/pixdesc.h"
#include "libavutil/avstring.h"
#include "libavutil/imgutils.h"
#include "libavcodec/avcodec.h"
#include "libavfilter/avfilter.h"
#include "libavfilter/buffersrc.h"
#include "libavfilter/buffersink.h"
}

using namespace std;


class CH264Decoder
{
public:
    CH264Decoder();
    ~CH264Decoder();
    /*************************************************
      Function:initial
      Description:初始化
      Input:无
      Output:无
      Return:错误代码
      Others:无
    *************************************************/
    int initial();
    int initial2();
    
    /*************************************************
      Function:decode
      Description:解码
      Input:pDataIn-待解码数据，nInSize-待解码数据长度
      Output:pDataOut-解码后的数据，nWidth-解码后的图像宽度，nHeight-解码后的图像高度
      Return:错误代码
      Others:解码后的数据为RGB16格式
    *************************************************/
    int test();
    int decode(uint8_t *pDataIn, int nInSize, uint8_t *pDataOut, int *nWidth, int *nHeight);
    int decode3(uint8_t *pDataIn, int nInSize, uint8_t *pDataOut, int *nWidth, int *nHeight);
    
    /*************************************************
      Function:unInitial
      Description:销毁
      Input:无
      Output:无
      Return:无
      Others:无
    *************************************************/
    void unInitial();

private:
    void deleteYUVTab();
    void createYUVTab_16();
    void displayYUV_16(unsigned int *pdst, unsigned char *y, unsigned char *u,unsigned char *v,
                       int width, int height, int src_ystride, int src_uvstride, int dst_ystride);
private:
    int *colortab;
    int *u_b_tab;
    int *u_g_tab;
    int *v_g_tab;
    int *v_r_tab;

    unsigned int *rgb_2_pix;
    unsigned int *r_2_pix;
    unsigned int *g_2_pix;
    unsigned int *b_2_pix;

    AVCodec *codec;
    AVCodecContext *context;
    AVFrame *frame;
    AVPacket packet;
};



#endif // CH264DECODER_H

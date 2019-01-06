//
//  main.c
//  ffmpeg_part01
//
//  Created by lbb on 2019/1/6.
//  Copyright © 2019年 lbb. All rights reserved.
//

#include <stdio.h>
#include "config.h"
#include "libavutil/parseutils.h"

char *av_strdupll(const char *s)
{
    char *ptr = NULL;
    if (s) {
        size_t len = strlen(s) + 1;
        ptr = malloc(len);//av_realloc(NULL, len);
        if (ptr) {
            memset(ptr, 0, len);
            memcpy(ptr, s, len);
        }
    }
    return ptr;
}



typedef struct {
    int width;
    int height;
} AVCodecContext;

int main(int argc, const char * argv[]) {
    // insert code here...
    const char *uri = "?cafile=file1";
    char buf[1024];
    const char *p = strchr(uri, '?');
    if (!p)
        return 0 ;
    
    char *ca_file;
    if ( av_find_info_tag(buf, sizeof(buf), "cafile", p))
        ca_file = av_strdupll(buf);
    
    printf("%s\n",ca_file);
    
    AVCodecContext context;
    AVCodecContext *dec = &context;
    
    char *canvas_size = "qntsc";
    if (canvas_size &&
        av_parse_video_size(&dec->width, &dec->height, canvas_size) < 0) {
        
    }
    
    return 0;
}

//
//  SpsDecode.h
//  PLRecorderKit
//
//  Created by 0day on 14/11/3.
//  Copyright (c) 2014年 qgenius. All rights reserved.
//

#ifndef __PLRecorderKit__SpsDecode__
#define __PLRecorderKit__SpsDecode__

#include <stdio.h>
#include <math.h>

typedef unsigned int UINT;
typedef unsigned char BYTE;
typedef unsigned long DWORD;

UINT Ue(BYTE *pBuff, UINT nLen, UINT &nStartBit);


int Se(BYTE *pBuff, UINT nLen, UINT &nStartBit);


DWORD u(UINT BitCount,BYTE * buf,UINT &nStartBit);


bool h264_decode_sps(BYTE * buf,unsigned int nLen,int &width,int &height);

#endif /* defined(__PLRecorderKit__SpsDecode__) */

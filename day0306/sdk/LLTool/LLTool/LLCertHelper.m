//
//  LLCertHelper.m
//  LLTool
//
//  Created by Mac on 2019/1/15.
//  Copyright © 2019 BaiRuiTechnology. All rights reserved.
//

#import "LLCertHelper.h"

@implementation LLCertHelper

+ (void)test
{
    SecKeyRef           publicKeyRef = nil;
    SecCertificateRef   cert = nil;
    
    NSLog(@"testtesttesttesttest");
    
//#if __IPHONE_OS_VERSION_MAX_ALLOWED >= __IPHONE_12_0
//    publicKeyRef = SecCertificateCopyKey(cert);
//#else
//    NSLog(@"< iOS 12");
//#endif

    //if ([[UIDevice currentDevice].systemVersion floatValue] >= 12)

    if (@available(iOS 12.0, *))
    {
        publicKeyRef = SecCertificateCopyKey(cert);
    } else {
        publicKeyRef = SecCertificateCopyPublicKey(cert);
    }
}
@end

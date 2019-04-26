//
//  staticlib01.m
//  staticlib01
//
//  Created by Mac on 2019/4/26.
//  Copyright © 2019 BaiRuiTechnology. All rights reserved.
//

#import "staticlib01.h"

@implementation staticlib01

+ (instancetype)sharedinstance {
    static staticlib01 *s_instance;
    static dispatch_once_t onceToken;
    dispatch_once(&onceToken, ^{
        s_instance = [[self alloc] init];
    });
    
    return s_instance;
}

- (instancetype)init {
    if (self = [super init]) {

        
    }
    return self;
}

- (void)test
{
    NSLog(@"%s",__func__);
}

@end

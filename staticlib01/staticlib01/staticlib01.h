//
//  staticlib01.h
//  staticlib01
//
//  Created by Mac on 2019/4/26.
//  Copyright © 2019 BaiRuiTechnology. All rights reserved.
//

#import <Foundation/Foundation.h>

@interface staticlib01 : NSObject

+ (instancetype)sharedinstance;

- (void)test;

@end

//
//  UIView+Test.m
//  AnyChatXingYe
//
//  Created by Mac on 2019/4/17.
//  Copyright © 2019 BaiRuiTechnology. All rights reserved.
//

#import "UIView+Test.h"
#import <objc/runtime.h>

@implementation UIView (Test)

+ (BOOL)resolveInstanceMethod:(SEL)sel {
    return NO;
}

- (id)forwardingTargetForSelector:(SEL)aSelector {
    NSString *sel = NSStringFromSelector(aSelector);
    
//    if ([sel isEqualToString:@"setImage:"] || [sel containsString:@"image"])
    {
        //1
//        return [UIImageView new];
        //2
        Class class = objc_allocateClassPair(NSClassFromString(@"NSObject"),"AvoidCrashTarget",0);
        class_addMethod(class, aSelector, class_getMethodImplementation([self class], @selector(avoidCrashAction)), "@@:");
//        //3
//        NSString* name = @"AvoidCrashTarget";
//        Class class = NSClassFromString(name);
//        if (!class) {
//            //alloc class pair
//            class = objc_allocateClassPair(NSClassFromString(@"NSObject"), name.UTF8String, 0);
//            //never try to add ivar to the new class, or you will find place using a wrong map.
//
//            //register the class
//            //
//            objc_registerClassPair(class);
//            class_addMethod(class, aSelector, class_getMethodImplementation([self class], @selector(avoidCrashAction)), "@@:");
//        }
        
        id tempObject = [[class alloc] init];
        return tempObject;
    }
    
    return [super forwardingTargetForSelector:aSelector];
}

- (NSObject *)avoidCrashAction {
    return nil;
}

@end

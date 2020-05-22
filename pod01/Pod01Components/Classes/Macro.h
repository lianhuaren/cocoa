//
//  Macro.h
//  LLTool
//
//  Created by Mac on 2019/3/6.
//  Copyright © 2019 aaaTechnology. All rights reserved.
//

#ifndef Macro_h
#define Macro_h
//获取屏幕 宽度、高度
#define SCREEN_WIDTH ([UIScreen mainScreen].bounds.size.width)
#define SCREEN_HEIGHT ([UIScreen mainScreen].bounds.size.height)

#define WS(weakSelf)  __weak __typeof(&*self)weakSelf = self

#define FONTS(size) [UIFont systemFontOfSize:(size)]

// 获取RGB颜色
#define RGBA(r,g,b,a) [UIColor colorWithRed:r/255.0f green:g/255.0f blue:b/255.0f alpha:a]
#define RGB(r,g,b) RGBA(r,g,b,1.0f)
// rgb颜色转换（16进制->10进制）
#define UIColorFromRGB(rgbValue) [UIColor colorWithRed:((float)((rgbValue & 0xFF0000) >> 16))/255.0 green:((float)((rgbValue & 0xFF00) >> 8))/255.0 blue:((float)(rgbValue & 0xFF))/255.0 alpha:1.0]
#define UIColorFromRGBA(rgbValue,a)        [UIColor colorWithRed:((float)((rgbValue & 0xFF0000) >> 16))/255.0f green:((float)((rgbValue & 0xFF00) >> 8))/255.0f blue:((float)(rgbValue & 0xFF))/255.0f alpha:a]

#define NavBar_StatusBar_Height   [UIApplication sharedApplication].statusBarFrame.size.height
#define NavBar_Height  (NavBar_StatusBar_Height + 44.0)  //状态栏高度+导航栏

#define AC_PLUS_HEIGHT 736.0

#define AC_PLUS_WIDTH  414.0

#define AC_WIDTH_FIT(width) (((width)/AC_PLUS_WIDTH)*SCREEN_WIDTH)

#endif /* Macro_h */

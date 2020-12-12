//
//  ViewController.h
//  GLPaint
//
//  Created by jiaguanglei on 15/12/14.
//  Copyright © 2015年 roseonly. All rights reserved.
//

#import <UIKit/UIKit.h>


//导入网络框架
#import <AsyncNetwork/AsyncNetwork.h>

/** 需要遵循网络框架的协议*/
@interface ServerViewController : UIViewController <AsyncServerDelegate>
/** 服务器端*/
@property (retain) AsyncServer *server; // the server
/** 服务器名称，用于bonjour发现*/
@property (retain) NSString *serviceName; // bonjour service name
/** bonjour发现服务类型*/
@property (retain) NSString *serviceType; // bonjour service type
/** 监听端口，0为自动设置*/
@property (assign) NSUInteger listenPort; // listening port (0: automatic)


@end


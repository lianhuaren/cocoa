//
//  ClientViewController.h
//  GLPaint
//
//  Created by 新界教育 on 16/4/5.
//  Copyright © 2016年 roseonly. All rights reserved.
//

#import <UIKit/UIKit.h>

//导入网络框架
#import <AsyncNetwork/AsyncNetwork.h>

/** 需要遵循网络框架的协议*/
@interface ClientViewController : UIViewController <AsyncClientDelegate>
/** 客户端*/
@property (retain) AsyncClient *client; // client
/** bonjour发现服务类型*/
@property (retain) NSString *serviceType; // bonjour service type

@end


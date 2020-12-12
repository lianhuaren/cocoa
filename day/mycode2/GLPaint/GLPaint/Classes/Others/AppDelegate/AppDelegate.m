//
//  AppDelegate.m
//  GLPaint
//
//  Created by jiaguanglei on 15/12/14.
//  Copyright © 2015年 roseonly. All rights reserved.
//

#import "AppDelegate.h"

//导入服务器端
#import "ServerViewController.h"



@interface AppDelegate ()

@end

@implementation AppDelegate


- (BOOL)application:(UIApplication *)application didFinishLaunchingWithOptions:(NSDictionary *)launchOptions {
    //配置新服务器端
    [self newServer:nil];
    
    return YES;
}

//创建一个新的服务器端
- (void)newServer:(id)sender;
{
    //服务器端ID(递增)
    static NSUInteger serverId = 0;
    static NSString *name = @"AsyncServer";
    
    //创建并配置服务器端控制器
    ServerViewController *serverViewController = [[ServerViewController alloc] init];
    serverViewController.serviceType = @"_ClientServer._tcp";
    
    serverViewController.serviceName = [NSString stringWithFormat:@"%@ %ld", name, ++serverId];

    self.window = [[UIWindow alloc] initWithFrame:PP_SCREEN_RECT];
    self.window.rootViewController = [[UINavigationController alloc] initWithRootViewController:serverViewController];
    [self.window makeKeyAndVisible];
}


- (void)applicationWillResignActive:(UIApplication *)application {
    // Sent when the application is about to move from active to inactive state. This can occur for certain types of temporary interruptions (such as an incoming phone call or SMS message) or when the user quits the application and it begins the transition to the background state.
    // Use this method to pause ongoing tasks, disable timers, and throttle down OpenGL ES frame rates. Games should use this method to pause the game.
}

- (void)applicationDidEnterBackground:(UIApplication *)application {
    // Use this method to release shared resources, save user data, invalidate timers, and store enough application state information to restore your application to its current state in case it is terminated later.
    // If your application supports background execution, this method is called instead of applicationWillTerminate: when the user quits.
}

- (void)applicationWillEnterForeground:(UIApplication *)application {
    // Called as part of the transition from the background to the inactive state; here you can undo many of the changes made on entering the background.
}

- (void)applicationDidBecomeActive:(UIApplication *)application {
    // Restart any tasks that were paused (or not yet started) while the application was inactive. If the application was previously in the background, optionally refresh the user interface.
}

- (void)applicationWillTerminate:(UIApplication *)application {
    // Called when the application is about to terminate. Save data if appropriate. See also applicationDidEnterBackground:.
    
}

@end

//
//  ViewController.m
//  libdemo01
//
//  Created by Mac on 2019/4/26.
//  Copyright © 2019 aaaTechnology. All rights reserved.
//

#import "ViewController.h"
#import <staticlib01.h>

@interface ViewController ()

@end

@implementation ViewController

- (void)viewDidLoad {
    [super viewDidLoad];
    // Do any additional setup after loading the view, typically from a nib.
    
    [[staticlib01 sharedinstance] test];
    
}


@end

//
//  ViewController.m
//  pod01Example
//
//  Created by Mac on 2019/4/9.
//  Copyright © 2019 aaaTechnology. All rights reserved.
//

#import "ViewController.h"
#import <Pod01Components/MRAppDelegateComponents.h>



@interface ViewController ()

@end

@implementation ViewController

- (void)viewDidLoad {
    [super viewDidLoad];
    // Do any additional setup after loading the view, typically from a nib.
    
//    UIButton *doneButton = [UIButton buttonWithType:UIButtonTypeCustom];
    UIButton *doneButton = [UIButton buttonWithTitle:@"下一步" atTitleColor:[UIColor whiteColor] atTarget:self atAction:@selector(done)];
    [self.view addSubview:doneButton];
    
    doneButton.backgroundColor = [UIColor colorWithRed:3/255.0 green:139/255.0 blue:227/255.0 alpha:1];;
//    [doneButton setTitle:@"下一步" forState:UIControlStateNormal];
//    [doneButton setTitleColor:[UIColor whiteColor] forState:UIControlStateNormal];
    doneButton.titleLabel.font = [UIFont systemFontOfSize:13];
    doneButton.frame = CGRectMake((SCREEN_WIDTH-110)/2, 20, 110, 40);
    doneButton.layer.masksToBounds = YES;
    doneButton.layer.cornerRadius = 5;
//    [doneButton addTarget:self action:@selector(done) forControlEvents:UIControlEventTouchUpInside];

}

- (void)done
{
    NSLog(@">>>>>>%s",__FUNCTION__);
}



@end

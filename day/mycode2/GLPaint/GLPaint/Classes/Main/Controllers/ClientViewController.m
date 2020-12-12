//
//  ClientViewController.m
//  GLPaint
//
//  Created by 新界教育 on 16/4/5.
//  Copyright © 2016年 roseonly. All rights reserved.
//

#import "ClientViewController.h"
//S
#import "RSColorPickerView.h"
#import "RSColorFunctions.h"
#import "PaintView.h"
#import "Masonry.h"
#import "RSBrightnessSlider.h"
#import "RSOpacitySlider.h"
#import "Line.h"
#import "RSColorPickerView.h"
#import "RSColorFunctions.h"
#import "PaintView.h"
//E

/** 宏定义*/
/** 左侧页边距*/
#define LEFT_MARGIN 20.f
/** 调色盘边长*/
#define PALLET_LENTH_OF_SIDE 150.f

@interface ClientViewController () <RSColorPickerViewDelegate>

/** 调色盘*/
@property (nonatomic, strong) RSColorPickerView *colorPicker;
/** 亮度*/
@property (nonatomic, strong) RSBrightnessSlider *brightnessSlider;
/** 不透明度*/
@property (nonatomic, strong) RSOpacitySlider *opacitySlider;
/** 线宽*/
@property (nonatomic, strong) UISlider *lineWidthSlider;
/** 调色面板视图*/
@property (nonatomic, strong) UIView *optionView;
/** 用于记录是否开启了调色面板*/
@property (nonatomic, assign, getter=isOpened) BOOL open;
/**  线段数组 ***/
@property (nonatomic, strong) NSMutableArray *lines;
/** 声明一个绘图面板属性*/
@property (nonatomic, strong) PaintView *paintView;

@end

@implementation ClientViewController

- (void)viewDidLoad {
    [super viewDidLoad];
    //设置背景色
    self.view.backgroundColor = [UIColor lightGrayColor];
    // 1. 配置选项视图
    [self configOptionView];
    // 2. 创建colorPiker
    [self setupColorPicker];
    // 3. 创建画板
    [self setupPaintView];
    // 4. 撤销
    [self setupNavigationItem];
    // 5. 添加手势
    [self configGestureRecognizer];
    // 6. 注册KVO

    // 7. 配置网络服务
    [self configAsyncNetwork];
}

- (void)configOptionView
{
    self.optionView = [UIView new];
    self.optionView.backgroundColor = [UIColor clearColor];
    [self.view addSubview:self.optionView];
    [self.optionView mas_makeConstraints:^(MASConstraintMaker *make) {
        make.top.mas_equalTo(TOP_LAYOUT_HEIGHT);
        make.left.right.equalTo(self.view);
        make.bottom.mas_equalTo(TOP_LAYOUT_HEIGHT+PALLET_LENTH_OF_SIDE);
    }];
    
}

- (void)configGestureRecognizer
{
    UILongPressGestureRecognizer *longPressGR = [[UILongPressGestureRecognizer alloc] initWithTarget:self action:@selector(longPress:)];
    longPressGR.minimumPressDuration = .5f;
    longPressGR.numberOfTouchesRequired = 2;
    [_paintView addGestureRecognizer:longPressGR];
}



- (void)longPress:(UILongPressGestureRecognizer *)longPressGR
{
    if (longPressGR.state == UIGestureRecognizerStateBegan) {
        
        if (self.isOpened) {
            [_paintView mas_updateConstraints:^(MASConstraintMaker *make) {
                make.left.equalTo(self.view);
                make.right.equalTo(self.view);
                make.bottom.equalTo(self.view);
                make.top.mas_equalTo(TOP_LAYOUT_HEIGHT);
            }];
            
        } else {
            [_paintView mas_updateConstraints:^(MASConstraintMaker *make) {
                make.left.equalTo(self.view);
                make.right.equalTo(self.view);
                make.bottom.equalTo(self.view).offset(PALLET_LENTH_OF_SIDE);
                make.top.mas_equalTo(TOP_LAYOUT_HEIGHT+PALLET_LENTH_OF_SIDE);
            }];
            
        }
        self.open = !self.open;
        
        [UIView animateWithDuration:0.3 animations:^{
            [self.view layoutIfNeeded];
        }];
    }
    
}




- (void)setupNavigationItem
{
    //    [[UIBarButtonItem appearance] setTintColor:[UIColor whiteColor]];
    self.navigationItem.leftBarButtonItem = [[UIBarButtonItem alloc] initWithBarButtonSystemItem:UIBarButtonSystemItemUndo target:self action:@selector(undoMethod)];
    self.navigationItem.rightBarButtonItem = [[UIBarButtonItem alloc] initWithBarButtonSystemItem:UIBarButtonSystemItemRedo target:self action:@selector(redoMethod)];
    
    UIButton *titleBtn = [UIButton buttonWithType:UIButtonTypeCustom];
    [titleBtn setTitleColor:[UIColor darkGrayColor] forState:UIControlStateNormal];
    [titleBtn setTitleColor:[UIColor lightGrayColor] forState:UIControlStateHighlighted];
    titleBtn.frame = CGRectMake(0, 0, 60, 44);
    [titleBtn setTitle:@"保存到相册" forState:UIControlStateNormal];
    [titleBtn addTarget:self action:@selector(saveImage) forControlEvents:UIControlEventTouchUpInside];
    self.navigationItem.titleView = titleBtn;
}

#pragma mark - 保存图片到相册
// 监听点击
- (void)saveImage
{
    // 获取照片
    UIImage *paintImage = [_paintView paintImage];
    //    LogRed(@"%@", paintImage);
    
    UIImageWriteToSavedPhotosAlbum(paintImage, self, @selector(imageSavedToPhotosAlbum:didFinishSavingWithError:contextInfo:), nil);
}

- (void)imageSavedToPhotosAlbum:(UIImage *)image didFinishSavingWithError:(NSError *)error contextInfo:(void *)contextInfo
{
    NSString *message = @"";
    if (!error) {
        message = @"成功保存到相册";
    }else
    {
        message = [error description];
    }
    LogRed(@"message is %@",message);
}


#pragma mark - 监听undo
- (void)undoMethod
{
    
    [_paintView undo];
}

- (void)redoMethod
{
    [_paintView redo];
}


#pragma mark 配置调色盘colorPiker
- (void)setupColorPicker
{
    
    self.colorPicker = [[RSColorPickerView alloc] initWithFrame:CGRectMake(0, 0, PALLET_LENTH_OF_SIDE, PALLET_LENTH_OF_SIDE)];
    [self.colorPicker setSelectionColor:RSRandomColorOpaque(YES)];
    
    
    [self.colorPicker setDelegate:self];
    self.colorPicker.cropToCircle = YES;
    [self.optionView addSubview:self.colorPicker];
    //初始化亮度值
    self.colorPicker.brightness = 1.f;
    //初始化不透明度值
    self.colorPicker.opacity = 1.f;
    
    //设置初始化颜色为黑色，如果不设置该颜色为黑色，初始颜色为随机
    //设置初始颜色为黑色后，亮度值将会改变为0
    self.colorPicker.selectionColor = [UIColor blackColor];
    
    // View that controls brightness
    self.brightnessSlider = [[RSBrightnessSlider alloc] initWithFrame:CGRectMake(PALLET_LENTH_OF_SIDE + 20, 10, SCREEN_WIDTH - PALLET_LENTH_OF_SIDE - 40, 20)];
    
    [self.brightnessSlider setColorPicker:self.colorPicker];
    [self.optionView addSubview:self.brightnessSlider];
    
    // View that controls opacity
    self.opacitySlider = [[RSOpacitySlider alloc] initWithFrame:CGRectMake(PALLET_LENTH_OF_SIDE + 20, 10+20+20, SCREEN_WIDTH - PALLET_LENTH_OF_SIDE - 40, 20)];
    
    
    [self.opacitySlider setColorPicker:self.colorPicker];
    [self.optionView addSubview:self.opacitySlider];
    
    self.lineWidthSlider = [[UISlider alloc] initWithFrame:CGRectMake(PALLET_LENTH_OF_SIDE + 20, 10+20+20+20+20, SCREEN_WIDTH - PALLET_LENTH_OF_SIDE - 40, 20)];
    self.lineWidthSlider.maximumValue = 15.f;
    self.lineWidthSlider.minimumValue = 1.f;
    //初始化线宽值
    self.lineWidthSlider.value = 4.f;
    self.paintView.lineWidth = self.lineWidthSlider.value;
    [self.optionView addSubview:self.lineWidthSlider];
    [self.lineWidthSlider addTarget:self action:@selector(lineWidthChanged:) forControlEvents:UIControlEventValueChanged];
}

#pragma mark 线宽发生了改变
- (void)lineWidthChanged:(id)sender
{
    if ([sender isKindOfClass:[UISlider class]]) {
        self.lineWidthSlider = sender;
    }
    [self.paintView setLineWidth:self.lineWidthSlider.value];
}

#pragma mark 调色板上的颜色发生变化时触发的方法
- (void)colorPickerDidChangeSelection:(RSColorPickerView *)colorPicker
{
    UIColor *color = [colorPicker selectionColor];
    [self.paintView setPaintColor:color];
    self.brightnessSlider.value = [colorPicker brightness];
    self.opacitySlider.value = [colorPicker opacity];
}



#pragma mark 配置画板PaintView
- (void)setupPaintView
{
    self.paintView = [PaintView paintView];
    
    
    self.paintView.backgroundColor = [UIColor whiteColor];
    
    // 设置线宽
    self.paintView.lineWidth = self.lineWidthSlider.value;
    
    [self.view addSubview:self.paintView];
    
    // 约束
    [self.paintView mas_makeConstraints:^(MASConstraintMaker *make) {
        
        make.left.equalTo(self.view);
        make.right.equalTo(self.view);
        make.bottom.equalTo(self.view);
        make.top.mas_equalTo(TOP_LAYOUT_HEIGHT);
    }];
    
    
    
}
/////////////////////////////////////////////////////////////////////
///////////////////////////////网络部分///////////////////////////
/////////////////////////////////////////////////////////////////////

- (void)dealloc {
    [self.client stop];
    self.client.delegate = nil;
}

//配置网络框架
- (void)configAsyncNetwork
{
    // 创建并配置客户端
    self.client = [AsyncClient new];
    self.client.serviceType = self.serviceType;
    self.client.delegate = self;
    
    // 开启客户端
    [self.client start];
    
    // 更新状态
    [self updateStatus];
    
}
#pragma mark 私有方法－更新状态
- (void)updateStatus;
{
    //显示连接的数量
    LogRed(@"Conntected to %ld server(s)", self.client.connections.count);
}

//发送消息给全部连接的服务器
- (void)sendInput:(id)sender;
{
    // ask the client to send the input string to all connected servers
    [self.client sendObject:@"客户端发送的字符串,tag=1"];
    
    // display log entry
    NSString *string = [NSString stringWithFormat:@">> %@\n", @"测试字符串,tag=1"];
    LogOrange(@"客户端发送%@", string);
    
    //清除输入
//    self.input.stringValue = @"";
}

#pragma mark - AsyncClientDelegate协议中的方法

- (void)client:(AsyncClient *)theClient didConnect:(AsyncConnection *)connection;
{
    //显示已经连接到哪个服务器
    NSString *string = [NSString stringWithFormat:@"[connected to %@]\n", connection.netService.name];
    LogBlue(@"客户端发送%@", string);
    
    // update the status
    [self updateStatus];
}

- (void)client:(AsyncClient *)theClient didDisconnect:(AsyncConnection *)connection;
{
    // display log entry
    NSString *string = [NSString stringWithFormat:@"[disconnected from %@]\n", connection.netService.name];
    LogBlue(@"客户端发送%@", string);
    
    // update the status
    [self updateStatus];
}

- (void)client:(AsyncClient *)theClient didReceiveCommand:(AsyncCommand)command object:(id)object connection:(AsyncConnection *)connection;
{
    // display log entry
    NSString *string = [NSString stringWithFormat:@"<< [%@] %@\n", connection.netService.name, object];
    LogPurple(@"服务器端接收客户端%@", string);
}

- (void)client:(AsyncClient *)theClient didFailWithError:(NSError *)error;
{
    //显示错误信息
    LogRed(@"error] %@\n", error.localizedDescription);
}



@end

2018-12-16 09:37:37.847313+0800 JPVideoPlayerDemo[11251:349532] 111---<AVAssetResourceLoadingRequest: 0x60400020bfe0, URL request = <NSMutableURLRequest: 0x60400020bb80> { URL: systemCannotRecognitionScheme:www.newpan.com }, request ID = 2, content information request = <AVAssetResourceLoadingContentInformationRequest: 0x60400020c1d0, content type = "(null)", content length = 0, byte range access supported = NO, disk caching permitted = YES, renewal date = (null)>,
 data request = <AVAssetResourceLoadingDataRequest: 0x600000202150, requested offset = 0, requested length = 2, requests all data to end of resource = NO, current offset = 0>>
[DEBUG] [Thread: 04] 111-开始网络请求, 网络请求创建一个 dataTask, id 是: 1,request:<NSMutableURLRequest: 0x60400020c2a0> { URL: http://127.0.0.1:8080/inweb01/a2.mp4 }

2018-12-16 09:37:37.894831+0800 JPVideoPlayerDemo[11251:349532] 111-444 indexString:{"com.newpan.size.key.www":2988434,"com.newpan.response.header.key.www":{"Content-Type":"video\/mp4","Last-Modified":"Sat, 10 Nov 2018 01:07:14 GMT","Content-Range":"bytes 0-1\/2988434","Accept-Ranges":"bytes","Date":"Sun, 16 Dec 2018 01:37:37 GMT","Content-Length":"2","Etag":"W\/\"2988434-1541812034000\""}}
2018-12-16 09:37:37.906197+0800 JPVideoPlayerDemo[11251:349532] 111-444 indexString:{"com.newpan.response.header.key.www":{"Content-Type":"video\/mp4","Last-Modified":"Sat, 10 Nov 2018 01:07:14 GMT","Content-Range":"bytes 0-1\/2988434","Accept-Ranges":"bytes","Date":"Sun, 16 Dec 2018 01:37:37 GMT","Content-Length":"2","Etag":"W\/\"2988434-1541812034000\""},"com.newpan.size.key.www":2988434,"com.newpan.zone.key.www":["{0, 2}"]}

2018-12-16 09:37:37.913150+0800 JPVideoPlayerDemo[11251:349532] 111---<AVAssetResourceLoadingRequest: 0x600000202390, URL request = <NSMutableURLRequest: 0x6000002035a0> { URL: systemCannotRecognitionScheme:www.newpan.com }, request ID = 4, content information request = (null),
 data request = <AVAssetResourceLoadingDataRequest: 0x60400020c2d0, requested offset = 0, requested length = 2988434, requests all data to end of resource = YES, current offset = 0>>
[DEBUG] [Thread: 06] 111-开始网络请求, 网络请求创建一个 dataTask, id 是: 2,request:<NSMutableURLRequest: 0x60400020c650> { URL: http://127.0.0.1:8080/inweb01/a2.mp4 }

2018-12-16 09:37:37.919177+0800 JPVideoPlayerDemo[11251:349532] 111-444 indexString:{"com.newpan.response.header.key.www":{"Content-Type":"video\/mp4","Last-Modified":"Sat, 10 Nov 2018 01:07:14 GMT","Content-Range":"bytes 2-2988433\/2988434","Accept-Ranges":"bytes","Date":"Sun, 16 Dec 2018 01:37:37 GMT","Content-Length":"2988432","Etag":"W\/\"2988434-1541812034000\""},"com.newpan.size.key.www":2988434,"com.newpan.zone.key.www":["{0, 2}"]}
2018-12-16 09:37:37.930854+0800 JPVideoPlayerDemo[11251:349532] 111-444 indexString:{"com.newpan.response.header.key.www":{"Content-Type":"video\/mp4","Last-Modified":"Sat, 10 Nov 2018 01:07:14 GMT","Content-Range":"bytes 2-2988433\/2988434","Accept-Ranges":"bytes","Date":"Sun, 16 Dec 2018 01:37:37 GMT","Content-Length":"2988432","Etag":"W\/\"2988434-1541812034000\""},"com.newpan.size.key.www":2988434,"com.newpan.zone.key.www":["{0, 2988434}"]}

2018-12-16 09:37:37.943377+0800 JPVideoPlayerDemo[11251:349532] 111---<AVAssetResourceLoadingRequest: 0x6000002035e0, URL request = <NSMutableURLRequest: 0x6000002036a0> { URL: systemCannotRecognitionScheme:www.newpan.com }, request ID = 6, content information request = (null),
 data request = <AVAssetResourceLoadingDataRequest: 0x60400020c7b0, requested offset = 2, requested length = 2988432, requests all data to end of resource = YES, current offset = 2>>

2018-12-16 09:37:37.945937+0800 JPVideoPlayerDemo[11251:349532] 111-333---didCancelLoadingRequest:<AVAssetResourceLoadingRequest: 0x6000002035e0, URL request = <NSMutableURLRequest: 0x6000002036a0> { URL: systemCannotRecognitionScheme:www.newpan.com }, request ID = 6, content information request = (null),
 data request = <AVAssetResourceLoadingDataRequest: 0x60400020c7b0, requested offset = 2, requested length = 2988432, requests all data to end of resource = YES, current offset = 720898>>

2018-12-16 09:37:38.067413+0800 JPVideoPlayerDemo[11251:349532] 111---<AVAssetResourceLoadingRequest: 0x600000203800, URL request = <NSMutableURLRequest: 0x6000002037a0> { URL: systemCannotRecognitionScheme:www.newpan.com }, request ID = 8, content information request = (null),
 data request = <AVAssetResourceLoadingDataRequest: 0x600000203940, requested offset = 65538, requested length = 2922896, requests all data to end of resource = YES, current offset = 65538>>


@property (strong, nonatomic) NSURLSessionDataTask *dataTask;

@property (strong, nonatomic) NSURLRequest *request;

mvhd(Time scale, Duration)
0000 0258 (600)
0000 d539 (54585) /600 = 90.975

tkhd(宽度, 高度)
02 7c00 00 (636)
01 6800 00 (360)

mdhd(Time scale, Duration)
00 005d c0 (24000)
00 214c 24 (2182180) /24000 = 90.924

hdlr
76 6964 65 (vide)

vmhd
766d 6864 (vmhd)

stbl
7374 626c (stbl)

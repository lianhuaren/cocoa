//
//  AudioProvider.swift
//  swiftdemo2
//
//  Created by 佰锐 on 2019/10/29.
//  Copyright © 2019 admin. All rights reserved.
//

import Foundation
import RxSwift

public struct AudioProvider {}

extension AudioProvider {
    public static func fetchAudioBanners() -> Observable<[MusicSheetInfo]> {
        let ballad = MusicSheetInfo()
        ballad.type = 23
        ballad.title = "情歌对唱榜"
        ballad.imgUrl = "https://timgsa.baidu.com/timg?image&quality=80&size=b9999_10000&sec=1554616722369&di=cdfdf3fc4c951c44e40b9d63cad0a2a9&imgtype=0&src=http%3A%2F%2Fi1.hdslb.com%2Fbfs%2Farchive%2F82fe4556b587af3350ff80d56bf803eac661d75f.jpg"
        
        let video = MusicSheetInfo()
        video.type = 24
        video.title = "影视金曲榜"
        video.imgUrl = "https://timgsa.baidu.com/timg?image&quality=80&size=b9999_10000&sec=1554618318731&di=88c00b69e53bd3435c89a1193aec0ab7&imgtype=0&src=http%3A%2F%2Fb-ssl.duitang.com%2Fuploads%2Fitem%2F201404%2F23%2F20140423102400_LQEFH.thumb.700_0.jpeg"
        
        let net = MusicSheetInfo()
        net.type = 25
        net.title = "网络歌曲"
        net.imgUrl = "https://timgsa.baidu.com/timg?image&quality=80&size=b9999_10000&sec=1554618360373&di=f921a927689004dacc3461b3329ede95&imgtype=0&src=http%3A%2F%2Fimg15.3lian.com%2F2015%2Ff2%2F63%2Fd%2F93.jpg"
        return Observable.just([ballad, video, net])
    }
}

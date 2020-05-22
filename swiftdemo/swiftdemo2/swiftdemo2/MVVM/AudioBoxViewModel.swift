//
//  AudioBoxViewModel.swift
//  swiftdemo2
//
//  Created by aaa on 2019/10/29.
//  Copyright © 2019 admin. All rights reserved.
//

import Foundation
import RxSwift
import RxSwiftExt

protocol AudioBoxViewModelInputs {
    func beginRefreshing()
}

protocol AudioBoxViewModelOutputs {
    var bannerLoaded: Observable<[MusicSheetInfo]> { get }
}

protocol AudioBoxViewModelType {
    var inputs: AudioBoxViewModelInputs { get }
    var outputs: AudioBoxViewModelOutputs { get }
}

class AudioBoxViewModel: AudioBoxViewModelType
    ,AudioBoxViewModelInputs
,AudioBoxViewModelOutputs {
    var inputs: AudioBoxViewModelInputs {
        return self
    }
    
    var outputs: AudioBoxViewModelOutputs {
        return self
    }
    
    init() {
        let refresh = refreshRelay
        .asObservable()
        .share()
        
        let bannerReq = refresh
            .flatMap {
                AudioProvider
                .fetchAudioBanners()
                .materialize()
            }.share()
        bannerLoaded = bannerReq.elements()
        
//        bannerLoaded = refresh.flatMap({
//            AudioProvider
//                .fetchAudioBanners()
//        })
        print(bannerLoaded)
    }
    
    fileprivate let refreshRelay = PublishSubject<Void>()
    func beginRefreshing() {
        refreshRelay.onNext(Void())
    }
    
    var bannerLoaded: Observable<[MusicSheetInfo]>
    
    
}

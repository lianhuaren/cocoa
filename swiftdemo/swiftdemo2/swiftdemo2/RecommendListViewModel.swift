//
//  RecommendListViewModel.swift
//  swiftdemo2
//
//  Created by 佰锐 on 2019/10/29.
//  Copyright © 2019 admin. All rights reserved.
//

import Foundation
import UIKit
import RxSwift

class RecommendListViewModel: NSObject {
    let data = Variable<[GroupedJHSection]>([])
    
    let refreshEnd = PublishSubject<Void>()
    
    var json:String!
    
    override init() {
        super.init()
    }
    
    func reload(type:String, json:String) {
        let api = String.jhRequest(type: type, json: json)
        
        bsLoadingProvider
            .rx.request(api)
            .asObservable().mapModel(JHRecommendListEntity.self)
            .showErrorToast()
            .subscribe(onNext: { [weak self] (model) in
                print("reload 11")
                
                let jsonprefix:String = model.info?.np ?? ""
                let jsonStr = String.init(format: "%@-20.json", jsonprefix)
                self?.json = jsonStr
                
                let arr:[JHListEntity] = model.list ?? []
                var seconArr:[GroupedJHSection] = []
                
                for entity in arr {
                    print(entity)
                    let con:[Top_commentsEntity] = entity.top_comments ?? []
                    seconArr.append(GroupedJHSection(header: entity, items: con))
                }
                
                if json == "0-20.json" {
                    self?.data.value = seconArr
                } else {
                    self?.data.value += seconArr
                }
                
                self?.refreshEnd.onNext(())
                
            }, onError: { [weak self] (error) in
                print("reload 22")
                
                self?.refreshEnd.onNext(())
            })
    }
}

//
//  ObservableType+HandyJSON.swift
//  swiftdemo2
//
//  Created by 佰锐 on 2019/10/29.
//  Copyright © 2019 admin. All rights reserved.
//

import Foundation
import RxSwift
import Moya
import HandyJSON

extension Response {
    func mapModel<T: HandyJSON>(_ type: T.Type) -> T {
        let jsonString = String.init(data: data, encoding: .utf8)
        return JSONDeserializer<T>.deserializeFrom(json: jsonString)!
    }
}

extension ObservableType where E == Response {
    public func mapModel<T: HandyJSON>(_ type: T.Type) -> Observable<T> {
        return flatMap { (response) -> Observable<T> in
            return Observable.just(response.mapModel(T.self))
        }
    }
}

extension Observable {
    func showErrorToast() -> Observable<Element> {
        return self.do(onNext: { (response) in
            print("showErrorToast 11")
        }, onError: { (error) in
            print("showErrorToast 22")
        })
    }
}

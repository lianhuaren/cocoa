//
//  Observable+ObjectMapper.swift
//  TFBaseLib_Swift
//
//  Created by xiayiyong on 2017/3/23.
//  Copyright © 2017年 上海赛可电子商务有限公司. All rights reserved.
//

import Foundation
import Moya
import ObjectMapper
import RxSwift

extension PrimitiveSequence where TraitType == SingleTrait, ElementType == Response {
    
    // MARK: - object
    
    func mapToObject<T: Mappable>(type: T.Type) -> Observable<T?> {
        
        return mapToResponse()
            .map {
                
                guard let dict =  $0 as? [String:Any], dict.count > 0 else{
                    return nil
                }
                
                guard let obj = Mapper<T>().map(JSON: dict) else {
                    return nil
                }
                
                return obj
            }
    }
    
    // MARK: - array
//    let dictionary = $0 as? [String: AnyObject]
//
//                   guard let array = dictionary?["stories"] as? [[String: Any]] else{
//                       return []
//                   }
    func mapToArray<T: Mappable >(type: T.Type) -> Observable<[T]> {
        return mapToResponse()
            .map {
                guard let array = $0 as? [[String: Any]] else{
                    return []
                }
                
                let arr = Mapper<T>().mapArray(JSONArray: array)
                return arr
            }
    }

    // MARK: - response
    
    private func mapToResponse() -> Observable<Any?> {
        return asObservable().map { response in
            
            guard ((200...209) ~= response.statusCode) else {
                return nil
            }
            
            guard var json = try? JSONSerialization.jsonObject(with: response.data, options: .allowFragments) else {
                return nil
            }
            
            
            let jsonString = String.init(data: response.data, encoding: .utf8)
            if let data = jsonString?.data(using: .utf8) {
                do {
                    json = try JSONSerialization.jsonObject(with: data, options: []) as? [String: Any]
                } catch {
                    print(error.localizedDescription)
                }
            }
            
            return json
        }
    }
}

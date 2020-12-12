//
//  WebService.swift
//  swift01
//
//  Created by  on 2020/8/19.
//  Copyright © 2020 admin. All rights reserved.
//

import Foundation

final class WebService {
    func load<T>(resource: Resource<T>, completion:@escaping(T?)->Void) {
        URLSession.shared.dataTask(with: resource.url) {
            (receivedData, _, error) in
            if let error = error {
                print(error.localizedDescription)
                return;
            }
            guard let data = receivedData else {return completion(nil)}
            completion(resource.parse(data))
        }.resume()
    }
    

}

final class SqlService {
    func load<T>(resource: SqlResource<T>, completion:(([T]?)->Void)) {
        let sql = resource.sqlSelect()
        print(sql!)
        
        let object = [T]()
        completion(object)
    }
}













//
//  Resource.swift
//  swift01
//
//  Created by  on 2020/8/18.
//  Copyright © 2020 admin. All rights reserved.
//

import Foundation

struct Resource<T> {
    let url: URL
    let parse: (Data) -> T?
    
    
}

extension Resource where T: Decodable {
//    init(_ url: URL, parseJSON: @escaping (Data) -> T?) {
//        self.url = url
////        self.parse = { data in
////            let decoder = JSONDecoder()
////            let object = try? decoder.decode(T.self, from: data)
////            return object
////        }
//        self.parse = parseJSON
//
//
//    }
    init(_ url: URL) {
        self.url = url
        self.parse = Resource.commonParse
    }
    static func commonParse(_ data:Data) -> T? {
        let decoder = JSONDecoder()
        let object = try? decoder.decode(T.self, from: data)
        return object
    }

}

let sqlSelectPrefix = "select * from "

struct SqlResource<T> {
    let tablename: String
    let sqlSelect: () -> String?
}

extension SqlResource {
//    init(_ tablename: String, sqlSelect: @escaping () -> String?) {
//        self.tablename = tablename
//        self.sqlSelect = sqlSelect//SqlResource.commonSqlSelect;
////        self.retrieve = {
////            let sql = "select * from" + tablename
////            print("===sql:",sql)
////            let object = [T]()
////            return object
////        }
//    }
    func commonSqlSelect() -> String {
        return sqlSelectPrefix + self.tablename
    }
    
}

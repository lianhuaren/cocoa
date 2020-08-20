//
//  JSONPlaceholder.swift
//  swift01
//
//  Created by  on 2020/8/18.
//  Copyright © 2020 admin. All rights reserved.
//

import Foundation

struct JSONPlaceholder {
    static let url = URL(string: "https://jsonplaceholder.typicode.com/posts")
    
    let userID : Int
    let id : Int
    let title : String
    let body : String
    
    enum Keys: String, CodingKey {
        case userId
        case id
        case title
        case body
    }
    
}

extension JSONPlaceholder : Decodable {
    init(from decoder: Decoder) throws {
        let values = try decoder.container(keyedBy: Keys.self)
        userID = try values.decode(Int.self, forKey: .userId)
        title = try values.decode(String.self, forKey: .title)
        id = try values.decode(Int.self, forKey: .id)
        body = try values.decode(String.self, forKey: .body)

    }
}

extension JSONPlaceholder {
//    static let all = Resource<[JSONPlaceholder]>(JSONPlaceholder.url!) {
//        (data) -> [JSONPlaceholder]? in
//        let posts = try? JSONDecoder().decode([JSONPlaceholder].self, from: data)
//        return posts
//
//    }
    
//    static let all = Resource<[JSONPlaceholder]>(url:JSONPlaceholder.url!) {
//        (data) -> [JSONPlaceholder]? in
//        return Resource.commonParse(data);
//    }
    
    static let all = Resource<[JSONPlaceholder]>(url:JSONPlaceholder.url!,parse:Resource.commonParse)
    
}

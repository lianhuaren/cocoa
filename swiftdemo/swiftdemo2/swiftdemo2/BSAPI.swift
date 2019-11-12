//
//  BSAPI.swift
//  swiftdemo2
//
//  Created by 佰锐 on 2019/10/29.
//  Copyright © 2019 admin. All rights reserved.
//

import Foundation
import RxSwift
import Moya

enum APIManager {
    case submenus //精华目录
    case jhrecommend(json:String) /// 精华推荐列表
    case jhimage(json:String) /// 精华图片专区列表
    case jhvideo(json:String) /// 精华视频专区列表
    case jhremen(json:String) /// 精华排行专区列表
    case jhjoke(json:String) /// 精华笑话专区列表
    case detailCommentList(id:String,page:String , json:String) /// 详情评论列表
}

extension String {
    static func jhRequest(type:String, json:String) -> APIManager {
        switch type {
        case "推荐":
            return .jhrecommend(json: json)
        case "视频":
            return .jhvideo(json: json)
        case "图片":
            return .jhimage(json: json)
        case "笑话":
            return .jhjoke(json: json)
        case "排行":
            return .jhremen(json: json)
        default:
            return .jhrecommend(json: json)
        }
    }
}

extension APIManager: TargetType {
    var baseURL: URL {
        switch self {
        case .submenus:
            return URL.init(string: "http://s.budejie.com")!
        case .detailCommentList:
            return URL.init(string: "http://c.api.budejie.com")!
        default:
            return URL.init(string: "http://s.budejie.com")!
        }
    }
    
    var path: String {
        switch self {
        case .submenus: return "public/list-appbar/bs0315-iphone-4.5.9"
        case .jhimage(let json): return "topic/list/jingxuan/10/bs0315-iphone-4.5.9/\(json)"
        case .jhvideo(let json): return "topic/list/jingxuan/41/bs0315-iphone-4.5.9/\(json)"
        case .jhremen(let json): return "topic/list/remen/1/bs0315-iphone-4.5.9/\(json)"
        case .jhjoke(let json): return "topic/tag-topic/63674/hot/bs0315-iphone-4.5.9/\(json)"
        case .jhrecommend(let json): return "topic/list/jingxuan/1/bs0315-iphone-4.5.9/\(json)"
        case .detailCommentList(let id ,let page , let json): return "topic/comment_list/\(id)/\(page)/bs0315-iphone-4.5.9/\(json)"
        }
    }
    
    var method: Moya.Method {
        return .get
    }
    
    var sampleData: Data {
        return "".data(using: String.Encoding.utf8)!
    }
    
    var task: Task {
        let parameters = ["version": Bundle.main.infoDictionary!["CFBundleShortVersionString"]!]
        return .requestParameters(parameters: parameters, encoding: URLEncoding.default)
    }
    
    var headers: [String : String]? {
        return nil
    }
    
    
}

let LoadingPlugin = NetworkActivityPlugin { (type, target) in
    
}

let bsLoadingProvider = MoyaProvider<APIManager>()

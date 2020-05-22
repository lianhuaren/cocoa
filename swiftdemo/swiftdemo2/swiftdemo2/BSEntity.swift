//
//  BSEntity.swift
//  swiftdemo2
//
//  Created by aaa on 2019/10/29.
//  Copyright © 2019 admin. All rights reserved.
//

import Foundation
import HandyJSON
import RxDataSources

/// 个人信息
struct UEntity: HandyJSON {
    var header: [String]? //头像
    var uid:String? // uid
    var is_vip:Bool? // 是否会员
    var sex:String? //性别
    var name:String? //昵称
}


/// 置顶的评论
struct Top_commentsEntity: HandyJSON {
    var u: UEntity? // 个人信息
    var id: String?  //id
    var content:String? // 评论内容
    var passtime: String? //更新时间
    var like_count: String? //点赞人数
}

/// 视频
struct VideoEntity: HandyJSON {
    var video: [String]?
    var thumbnail: [String]? // 封面图
    var width: CGFloat?
    var height: CGFloat?
}

/// 图片
struct ImageEntity: HandyJSON {
    var big: [String]? // 封面图
    var width: CGFloat?
    var height: CGFloat?
}

/// 动态图
struct GifEntity: HandyJSON {
    var images: [String]?
    var gif_thumbnail: [String]? //封面图
    var width: CGFloat?
    var height: CGFloat?
}

struct JHListEntity: HandyJSON {
    var status:String?
    var cate:String?
    var name:String?
    var top_comments:[Top_commentsEntity]?
    var bookmark:String? //阅读数
    var text:String?
    var video:VideoEntity?
    var u:UEntity?
    var passtime:String?
    var type:String? //类型: text image video gif
    var id:String?
    var image:ImageEntity?
    var gif:GifEntity?
    var comment:String? //评论
    var up:String? //点赞
    var down:String? //踩
    var forward:String? //分享
//    var tags:[tagsEntity]? //社区类型
}

struct Info: HandyJSON {
    var count:String?
    var np:String? //上拉加载需要拼接的随机数据
}

/// 精华推荐列表
struct JHRecommendListEntity: HandyJSON{
    var info: Info?
    var list:[JHListEntity]?
}


struct GroupedJHSection {
    var header: JHListEntity?
    var items: [Item]
}

extension GroupedJHSection: SectionModelType {
    typealias Item = Top_commentsEntity
    
    init(original: Self, items: [Self.Item]) {
        self = original
        self.items = items
    }
}

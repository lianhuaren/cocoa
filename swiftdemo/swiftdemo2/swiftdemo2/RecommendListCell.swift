//
//  RecommendListCell.swift
//  swiftdemo2
//
//  Created by aaa on 2019/10/29.
//  Copyright © 2019 admin. All rights reserved.
//

import UIKit

class RecommendListCell: UITableViewCell {
    static func getCellHightData(data:Top_commentsEntity) -> CGFloat {
        return 60
//        let attributedText = self.init().contentAttributedText(data: data)
//        let hight = self.init().contextHight(attributedText: attributedText)
//        return hight + 8
    }
    
    func reloadData(data:Top_commentsEntity) -> Void {
        let content = String.init(format: "%@: %@", data.u?.name ?? "" , data.content ?? "")
        self.textLabel?.text = content
//        let attributedText = self.contentAttributedText(data: data)
//        let hight = self.contextHight(attributedText: attributedText)
//        labContent.attributedText = attributedText
//        bgView.frame = CGRect(x: 15, y: 0, width: kScreenWidth - 30, height: hight + 8)
//        labContent.frame = CGRect(x: 15, y: 8, width: kScreenWidth - 60, height: hight)
    }
}

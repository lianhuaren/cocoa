//
//  selectBar.swift
//  DYZB01
//
//  Created by temp on 2018/5/18.
//  Copyright © 2018年 temp. All rights reserved.
//

import UIKit

enum SelectBarStatus {
    case SelectBarStatusRecommend
    case SelectBarStatusGame
    case SelectBarStatusAmuse
    case SelectBarStatusFun
}

protocol SelectBarProtocol : class {
    func changeSelectBarItem(status : SelectBarStatus)
}

class selectBar: UIView {
    weak var delegate : SelectBarProtocol!
    var recommendBtn : UIButton!
    var gameBtn : UIButton!
    var amuseBtn : UIButton!
    var funBtn : UIButton!
    var bottomLine : UIView!
    
    
    override init(frame: CGRect) {
        super.init(frame: frame)
        
        recommendBtn = UIButton.init(frame: CGRect(x: 0, y: 0, width: frame.size.width / 4, height: 39))
        recommendBtn.setTitle("推荐", for: .normal)
        recommendBtn.addTarget(self, action: #selector(btnClick), for: .touchUpInside)
        recommendBtn.setTitleColor(UIColor.orange, for: .normal)
        recommendBtn.setTitleColor(UIColor.orange, for: .highlighted)
        recommendBtn.titleLabel?.font = UIFont.systemFont(ofSize: 15)
        addSubview(recommendBtn)
        
        gameBtn = UIButton.init(frame: CGRect(x: frame.size.width*1 / 4, y: 0, width: frame.size.width / 4, height: 39))
        gameBtn.setTitle("游戏", for: .normal)
        gameBtn.addTarget(self, action: #selector(btnClick), for: .touchUpInside)
        gameBtn.setTitleColor(UIColor.orange, for: .normal)
        gameBtn.setTitleColor(UIColor.orange, for: .highlighted)
        gameBtn.titleLabel?.font = UIFont.systemFont(ofSize: 15)
        addSubview(gameBtn)
        
        amuseBtn = UIButton.init(frame: CGRect(x: frame.size.width*2 / 4, y: 0, width: frame.size.width / 4, height: 39))
        amuseBtn.setTitle("娱乐", for: .normal)
        amuseBtn.addTarget(self, action: #selector(btnClick), for: .touchUpInside)
        amuseBtn.setTitleColor(UIColor.orange, for: .normal)
        amuseBtn.setTitleColor(UIColor.orange, for: .highlighted)
        amuseBtn.titleLabel?.font = UIFont.systemFont(ofSize: 15)
        addSubview(amuseBtn)
        
        funBtn = UIButton.init(frame: CGRect(x: frame.size.width*3 / 4, y: 0, width: frame.size.width / 4, height: 39))
        funBtn.setTitle("趣玩", for: .normal)
        funBtn.addTarget(self, action: #selector(btnClick), for: .touchUpInside)
        funBtn.setTitleColor(UIColor.orange, for: .normal)
        funBtn.setTitleColor(UIColor.orange, for: .highlighted)
        funBtn.titleLabel?.font = UIFont.systemFont(ofSize: 15)
        addSubview(funBtn)
        
        bottomLine = UIView.init(frame: CGRect(x: frame.size.width / 8-20, y: 39, width: 40, height: 1))
        bottomLine.backgroundColor = UIColor.orange
        addSubview(bottomLine)
    }
    
    required init?(coder aDecoder: NSCoder) {
        super.init(coder: aDecoder)
    }
    
    @objc func btnClick(sender: UIButton) {
        print(sender)
        
        recommendBtn.setTitleColor(UIColor.black, for: .normal)
        gameBtn.setTitleColor(UIColor.black, for: .normal)
        amuseBtn.setTitleColor(UIColor.black, for: .normal)
        funBtn.setTitleColor(UIColor.black, for: .normal)
     
        sender.setTitleColor(UIColor.orange, for: .normal)
    }
    
    
}

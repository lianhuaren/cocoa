//
//  ViewController.swift
//  DYZB01
//
//  Created by temp on 2018/5/18.
//  Copyright © 2018年 temp. All rights reserved.
//

import UIKit

class ViewController: UIViewController , SelectBarProtocol {
    func changeSelectBarItem(status: SelectBarStatus) {
        
    }
    

    override func viewDidLoad() {
        super.viewDidLoad()
        // Do any additional setup after loading the view, typically from a nib.
        navigationController?.navigationBar.barTintColor = UIColor.init(red: 247/255.5, green: 247/255.5, blue: 247/255.5, alpha: 1)
        view.backgroundColor = UIColor.white
        
        let bar = selectBar.init(frame: CGRect(x: 0, y: 110, width: ScreenWidth, height: 40))
        bar.delegate = self
        view.addSubview(bar)
    }

    override func didReceiveMemoryWarning() {
        super.didReceiveMemoryWarning()
        // Dispose of any resources that can be recreated.
    }


}


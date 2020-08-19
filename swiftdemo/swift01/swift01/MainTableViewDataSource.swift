//
//  MainTableViewDataSource.swift
//  swift01
//
//  Created by  on 2020/8/18.
//  Copyright © 2020 admin. All rights reserved.
//

import Foundation
import UIKit

class MainTableViewDataSource: NSObject {
    let webService = WebService()
    var dateMachines = [DateFmt]()
    
    fileprivate func setupDates() {
        for _ in 0...13 {
            let datefmt = DateFmt(date: Date())
            
            print("===",datefmt)
            dateMachines.append(datefmt)
            
            
        }
    }
    
    override init() {
        super.init()
        
        setupDates()
        getData()
    }
}

extension MainTableViewDataSource {
    func getData() {
        webService.load(resource: JSONPlaceholder.all) { [weak self] (posts) in
            guard let placeholder = posts else {return}
//            print(placeholder)
            _ = self?.convertPlaceholderToDreams(withPosts: placeholder)
            NotificationCenter.default.post(name: .dataRetrieved, object: nil)
        }
    }
    
    func convertPlaceholderToDreams(withPosts placeholders: [JSONPlaceholder]) -> [Dream] {
        return placeholders.map { placeholder in
            let randomNumber = 1
            let dateMachine = dateMachines[randomNumber]
            let dream = Dream(title: placeholder.title, dateString: dateMachine.dateString, tags:["happy, funny, sad, awesome"], description: placeholder.body)
            ApplicationData.DatabaseData.write(toData: dream)
            return dream
        }
    }
}

//
//  DreamData.swift
//  swift01
//
//  Created by  on 2020/8/19.
//  Copyright © 2020 admin. All rights reserved.
//

import Foundation

struct DreamData {
    var data = [Dream]()
}

extension DreamData {
    func write(toData data: Dream) {
        ApplicationData.DatabaseData.data.append(data)
    }
    
    func read() -> [Dream] {
        return ApplicationData.DatabaseData.data
    }
}

class ApplicationData {
    static var DatabaseData = DreamData()
}

//
//  DateFmt.swift
//  swift01
//
//  Created by  on 2020/8/18.
//  Copyright © 2020 admin. All rights reserved.
//

import Foundation

struct DateFmt {
    let dateString: String
    
    init(date: Date) {
        let formatter = DateFormatter()
        formatter.dateFormat = "d MMM yyyy"
        dateString = formatter.string(from: date)
    }
}

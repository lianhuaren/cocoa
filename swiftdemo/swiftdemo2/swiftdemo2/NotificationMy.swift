//
//  NotificationMy.swift
//  swiftdemo2
//
//  Created by aaa on 2019/10/28.
//  Copyright © 2019 admin. All rights reserved.
//

import Foundation

public final class ZY<Base> {
    public let base: Base
    
    public init(_ base: Base) {
        self.base = base
    }
}

public protocol ZYCompatible {
    associatedtype CompatibleType
    var zy: CompatibleType { get }
}

public extension ZYCompatible {
    public var zy: ZY<Self> {
        get {
            return ZY(self)
        }
    }
}

//extension NotificationCenter: ZYCompatible {}
//
//extension ZY where Base: NotificationCenter {
//    func post(name aName: NSNotification.Name, object anObject: Any?, userInfo aUserInfo: [AnyHashable : Any]? = nil) {
//        print("1111")
//        self.base.post(name: aName, object: anObject, userInfo: aUserInfo)
//    }
//
//}

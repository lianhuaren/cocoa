//
//  ExString.swift
//  Fate
//
//  Created by Archer on 2018/11/23.
//

import Foundation

fileprivate let blanks = ["NIL",
                          "Nil",
                          "nil",
                          "NULL",
                          "Null",
                          "null",
                          "(nil)",
                          "<nil>",
                          "(NULL)",
                          "(Null)",
                          "(null)",
                          "<NULL>",
                          "<Null>",
                          "<null>"]

public extension String {
    public var isBlank: Bool {
        return isEmpty ||
            self.trimmingCharacters(in: CharacterSet.whitespacesAndNewlines).isEmpty ||
            blanks.contains(self)
    }
    
    public var isNotBlank: Bool {
        return !isBlank
    }
    
    public var trimmed: String {
        return trimmingCharacters(in: CharacterSet.whitespacesAndNewlines)
    }
    
    public func filterBlank(_ valueOnBlank: String = StringOnNil) -> String {
        return isBlank ? valueOnBlank : self
    }
}

#if DEBUG
public let StringOnNil = "undefined"
#else
public let StringOnNil = ""
#endif

public extension Optional where Wrapped == String {
    public var isBlank: Bool {
        switch self {
        case .some(let value):
            return value.isBlank
        case .none:
            return true
        }
    }
    
    public var isNotBlank: Bool {
        return !isBlank
    }
    
    public var length: Int {
        switch self {
        case .some(let value):
            return value.count
        case .none:
            return 0
        }
    }
    
    public var trimmed: String? {
        switch self {
        case .some(let value):
            return value.trimmed
        case .none:
            return nil
        }
    }
    
    public func filterNil(_ valueOnNil: String = StringOnNil) -> String {
        switch self {
        case .some(let value):
            return value
        case .none:
            return valueOnNil
        }
    }
    
    public func filterBlank(_ valueOnBlank: String = StringOnNil) -> String {
        switch self {
        case .some(let value):
            return value.isBlank ? valueOnBlank : value
        case .none:
            return valueOnBlank
        }
    }
}

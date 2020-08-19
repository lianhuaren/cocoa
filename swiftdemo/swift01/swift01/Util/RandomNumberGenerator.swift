//
//  RandomNumberGenerator.swift
//  swift01
//
//  Created by  on 2020/8/19.
//  Copyright © 2020 admin. All rights reserved.
//

import Foundation
import GameplayKit

struct RandomNumberGenerator {
    fileprivate let source = GKRandomSource.sharedRandom()
    
    public func getRandomNumber(withUpperBound bound: Int) -> Double {
        let randomSource = source.nextInt(upperBound: bound)
        return Double(randomSource)
    }
}

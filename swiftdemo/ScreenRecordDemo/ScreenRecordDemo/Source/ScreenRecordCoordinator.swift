//
//  ScreenRecordCoordinator.swift
//  ScreenRecordDemo
//
//  Created by 佰锐 on 2019/9/29.
//  Copyright © 2019 admin. All rights reserved.
//

import Foundation
import UIKit

class ScreenRecordCoordinator: NSObject
{
    var viewOverlay : WindowUtil?
    let screenRecorder = ScreenRecorder()
    var recordCompleted:((Error?) -> Void)?
    
    override init()
    {
        super.init()
        
        viewOverlay = WindowUtil()
        viewOverlay?.onStopClick = { [weak self] in
            self?.stopRecording()
        }
    }
    deinit {
        print(String.init(format: "\(#function):%s", object_getClassName(self)))
    }

    func startRecording(withFileName fileName: String, recordingHandler: @escaping (Error?) -> Void,
                        onCompletion: @escaping (Error?)->Void)
    {
        self.recordCompleted = onCompletion
        self.viewOverlay?.show()
        
    }
    
    func stopRecording()
    {
        self.viewOverlay?.hide()
        self.recordCompleted?(nil)
    }
}

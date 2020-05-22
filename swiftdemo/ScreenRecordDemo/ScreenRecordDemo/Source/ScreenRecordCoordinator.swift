//
//  ScreenRecordCoordinator.swift
//  ScreenRecordDemo
//
//  Created by aaa on 2019/9/29.
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
        
        screenRecorder.startRecording(withFileName: fileName) { (error) in
            
        }
    }
    
    func stopRecording()
    {
        screenRecorder.stopRecording { (error) in
            self.viewOverlay?.hide()
            self.recordCompleted?(error)
        }
    }
}

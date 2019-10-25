//
//  WindowUtil.swift
//  ScreenRecordDemo
//
//  Created by 佰锐 on 2019/9/29.
//  Copyright © 2019 admin. All rights reserved.
//

import Foundation
import UIKit

protocol Overlayable
{
    func show()
    func hide()
}

class WindowUtil: NSObject, Overlayable
{
    var overlayWindow = UIWindow(frame: CGRect(x: 0, y: 0, width: UIScreen.main.bounds.width, height: 30))
    var stopButton = UIButton(type: UIButton.ButtonType.custom)
    var stopButtonColor = UIColor(red: 0.30, green: 0.67, blue: 0.99, alpha: 1.00)
    var onStopClick:(() -> ())?
    
    override init ()
    {
        super.init()
        self.setupViews()
    }
    
    
    deinit {
        print(String.init(format: "\(#function):%s", object_getClassName(self)))
    }

    func initViews()
    {
        let statusBarHeight = UIApplication.shared.statusBarFrame.height;
        let height = statusBarHeight+10
        overlayWindow = UIWindow(frame: CGRect(x: 0, y: 0, width: UIScreen.main.bounds.width, height: height))
        stopButton = UIButton(type: UIButton.ButtonType.custom)
    }
    
    func hide()
    {
        DispatchQueue.main.async {
            
            UIView.animate(withDuration: 0.3, animations: {
                self.stopButton.transform = CGAffineTransform(translationX: 0, y: -30)
            }, completion: { (animated) in
                self.overlayWindow.backgroundColor = .clear
                self.overlayWindow.isHidden = true
                self.stopButton.isHidden = true
                self.stopButton.transform = CGAffineTransform.identity
            })
            
        }
    }
    
    func setupViews ()
    {
        initViews()
        stopButton.setTitle("Stop Recording", for: .normal)
        stopButton.titleLabel?.font = UIFont.systemFont(ofSize: UIFont.smallSystemFontSize)
        
        stopButton.addTarget(self, action: #selector(stopRecording), for: UIControl.Event.touchDown)
        
        stopButton.frame = overlayWindow.frame
        overlayWindow.addSubview(stopButton)
        overlayWindow.windowLevel = UIWindow.Level(rawValue: CGFloat.greatestFiniteMagnitude)
        
    }
    
    @objc func stopRecording()
    {
        onStopClick?()
    }
    
    func show()
    {
        DispatchQueue.main.async {
            self.stopButton.transform = CGAffineTransform(translationX: 0, y: -30)
            self.stopButton.backgroundColor = self.stopButtonColor
            self.overlayWindow.makeKeyAndVisible()
            UIView.animate(withDuration: 0.3, animations: {
                self.stopButton.transform = CGAffineTransform.identity
            }, completion: { (animated) in
                
            })
        }
        
    }
}

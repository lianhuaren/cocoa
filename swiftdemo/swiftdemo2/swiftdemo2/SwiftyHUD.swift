//
//  SwiftyHUD.swift
//  swiftdemo2
//
//  Created by 佰锐 on 2019/10/28.
//  Copyright © 2019 admin. All rights reserved.
//

import Foundation
import MBProgressHUD

public struct SwiftyHUD {
    public static func show(message: String?,
                            duration: TimeInterval = 3,
                            textColor: UIColor = .white,
                            bezelAlpha: CGFloat = 0.6,
                            addedTo view: UIView? = UIApplication.shared.windows[0]) {
        guard let sourceView = view else { return }
        let hud = MBProgressHUD.showAdded(to: sourceView, animated: true)
        hud.isUserInteractionEnabled = false
        hud.bezelView.backgroundColor = UIColor.black.withAlphaComponent(bezelAlpha)
        hud.mode = .text
        hud.label.font = .systemFont(ofSize: 14)
        hud.label.textColor = textColor
        hud.label.numberOfLines = 0
        hud.label.text = message
        hud.margin = 8
        hud.hide(animated: true, afterDelay: duration)
        
    }
    
    
}


























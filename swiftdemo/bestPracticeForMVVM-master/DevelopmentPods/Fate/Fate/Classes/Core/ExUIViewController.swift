//
//  ExUIViewController.swift
//  Fate
//
//  Created by Archer on 2018/11/23.
//

import FDNamespacer

public extension FOLDin where Base : UIViewController {
//    public var fullNavbarHeight: CGFloat {
//        return navbarHeight + statusBarHeight
//    }

    public var navbarHeight: CGFloat {
        return base.navigationController?.navigationBar.frame.size.height ?? 44
    }

    public var statusBarHeight: CGFloat {
        var statusBarHeight = UIApplication.shared.statusBarFrame.size.height
        if #available(iOS 11.0, *) {
            // Account for the notch when the status bar is hidden
            statusBarHeight = max(UIApplication.shared.statusBarFrame.size.height, UIApplication.shared.delegate?.window??.safeAreaInsets.top ?? 0)
        }
        return statusBarHeight - extendedStatusBarDifference
    }

    // Extended status call changes the bounds of the presented view
    var extendedStatusBarDifference: CGFloat {
        return abs(base.view.bounds.height - (UIApplication.shared.delegate?.window??.frame.size.height ?? UIScreen.main.bounds.height))
    }
    
    public var tabBarOffset: CGFloat {
        // Only account for the tab bar if a tab bar controller is present and the bar is not translucent
        if let tabBarController = base.tabBarController, !(base.navigationController?.topViewController?.hidesBottomBarWhenPushed ?? false) {
            return tabBarController.tabBar.isTranslucent ? 0 : tabBarController.tabBar.frame.height
        }
        return 0
    }
    
    public func add(childViewController child: UIViewController, frame: CGRect = UIScreen.main.bounds) {
        guard !base.children.contains(child) else { return }
        child.view.frame = frame
        base.view.addSubview(child.view)
        base.addChild(child)
        child.didMove(toParent: base)
    }
    
    public func remove(childViewController child: UIViewController) {
        guard base.children.contains(child) else { return }
        child.willMove(toParent: nil)
        child.view.removeFromSuperview()
        child.removeFromParent()
    }
    
    @discardableResult
    public func popToViewController(advancedBy: Int, animated: Bool = true) -> Bool {
        assert(advancedBy < 0, "non positive")
        guard let navigationVc = base.navigationController else {
            NSLog("Fate warning: no navigation controller exists")
            return false
        }
        guard advancedBy < navigationVc.children.count else {
            NSLog("Fate warning: iligal advancedBy value")
            return false
        }
        let index = navigationVc.children.index(of: base)!
        let dstVc = navigationVc.children[index + advancedBy]
        navigationVc.popToViewController(dstVc, animated: animated)
        return true
    }
    
    @discardableResult
    public func popToViewController(metaType: UIViewController.Type, animated: Bool = true) -> Bool {
        guard let navigationVc = base.navigationController else {
            NSLog("Fate warning: no navigation controller exists")
            return false
        }
        guard let dstVc = navigationVc.children.last(where: { $0.isKind(of: metaType)  }) else {
            NSLog("Fate warning: no view controller(\(metaType)) exists in its navigation controller's stack")
            return false
        }
        navigationVc.popToViewController(dstVc, animated: animated)
        return true
    }
}

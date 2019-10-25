//
//  Identifiable.swift
//  Fate
//
//  Created by Archer on 2018/11/27.
//  Copyright © 2018年 Archer. All rights reserved.
//

import FDNamespacer
import UITableView_FDTemplateLayoutCell

/// 为UITableView/UICollectionView提供重用标
/// 识, 免去了硬编码reuseIdentifier的问题, 同
/// 时给dequeueXXX相关的方法提供了类型推断.
public protocol ReuseIdentifiable: class {
    static var reuseIdentifier: String { get }
}

public extension ReuseIdentifiable {
    /// 重用标识默认实现
    static var reuseIdentifier: String {
        return String(describing: self)
    }
}

// MARK: UITableView解决类型推断

extension UITableViewCell: ReuseIdentifiable {}
extension UITableViewHeaderFooterView: ReuseIdentifiable {}

public extension FOLDin where Base: UITableView {
    
    /// 给UITableView注册cell,
    /// 配合下面的dequeue方法取出cell.
    public func register<_Tp: UITableViewCell>(cellClass: _Tp.Type) {
        base.register(cellClass.self, forCellReuseIdentifier: cellClass.reuseIdentifier)
    }
    
    /// 从重用队列中取cell,
    /// cell必须先用上面的方法注册.
    public func dequeueReusableCell<_Tp: UITableViewCell>(for indexPath: IndexPath, cellClass: _Tp.Type = _Tp.self) -> _Tp {
        guard let cell = base.dequeueReusableCell(withIdentifier: cellClass.reuseIdentifier, for: indexPath) as? _Tp else {
            fatalError("Could not dequeue cell.")
        }
        return cell
    }
    
    /// 给UITableView注册headerFooterView,
    /// 配合下面的dequeue方法取出headerFooterView.
    public func register<_Tp: UITableViewHeaderFooterView>(headerFooterViewClass: _Tp.Type) {
        base.register(headerFooterViewClass.self, forHeaderFooterViewReuseIdentifier: headerFooterViewClass.reuseIdentifier)
    }
    
    /// 从重用队列中取headerFooterView,
    /// headerFooterView必须先用上面的方法注册.
    public func dequeueReusableHeaderFooterView<_Tp: UITableViewHeaderFooterView>(_ headerFooterViewClass: _Tp.Type = _Tp.self) -> _Tp? {
        guard let headerFooterView = base.dequeueReusableHeaderFooterView(withIdentifier: headerFooterViewClass.reuseIdentifier) as? _Tp? else {
            fatalError("Could not dequeue header/footer view.")
        }
        return headerFooterView
    }
}

// MARK: UICollectionView解决类型推断

extension UICollectionReusableView: ReuseIdentifiable {}

public extension FOLDin where Base: UICollectionView {
    
    /// 给UICollectionView注册cell,
    /// 配合下面的dequeue方法取出cell.
    public func register<_Tp: UICollectionViewCell>(_ cellClass: _Tp.Type) {
        base.register(cellClass.self, forCellWithReuseIdentifier: cellClass.reuseIdentifier)
    }
    
    /// 从重用队列中取cell,
    /// cell必须先用上面的方法注册.
    public func dequeueReusableCell<_Tp: UICollectionViewCell>(for indexPath: IndexPath, cellClass: _Tp.Type = _Tp.self) -> _Tp {
        guard let cell = base.dequeueReusableCell(withReuseIdentifier: cellClass.reuseIdentifier, for: indexPath) as? _Tp else {
            fatalError("Could not dequeue cell.")
        }
        return cell
    }
    
    /// 给UICollectionView注册supplementaryView,
    /// 配合下面的dequeue方法取出supplementaryView.
    public func register<_Tp: UICollectionReusableView>(_ supplementaryViewClass: _Tp.Type,
                                                        forSupplementaryViewOfKind elementKind: String) {
        base.register(supplementaryViewClass.self,
                      forSupplementaryViewOfKind: elementKind,
                      withReuseIdentifier: supplementaryViewClass.reuseIdentifier)
    }
    
    /// 从重用队列中取supplementaryView,
    /// supplementaryView必须先用上面的方法注册.
    public func dequeueReusableSupplementaryView<_Tp: UICollectionReusableView>(ofKind elementKind: String,
                                                                                for indexPath: IndexPath,
                                                                                supplementaryViewClass: _Tp.Type = _Tp.self) -> _Tp {
        guard let supplementaryView = base.dequeueReusableSupplementaryView(ofKind: elementKind,
                                                                            withReuseIdentifier: supplementaryViewClass.reuseIdentifier,
                                                                            for: indexPath) as? _Tp else {
                                                                                fatalError("Could not dequeue supplementary view.")
        }
        return supplementaryView
    }
}

// MARK: FDTemplateLayoutCell解决类型推断

public extension FOLDin where Base: UITableView {
    
    /// UITableView自动计算cell高度,
    /// cell必须先用上面的方法注册.
    public func heightForRowAt<_Tp: UITableViewCell>(_ indexPath: IndexPath,
                                                     cellClass: _Tp.Type,
                                                     configuration: @escaping (_Tp) -> Void) -> CGFloat {
        return base.fd_heightForCell(withIdentifier: cellClass.reuseIdentifier, cacheBy: indexPath, configuration: { (cell) in
            guard let cell = cell as? _Tp else {
                fatalError("Could not convert cell.")
            }
            configuration(cell)
        })
    }
    
    /// UITableView自动计算headerFooterView高度,
    /// cell必须先用上面的方法注册.
    public func heightForHeaderFooterView<_Tp: UITableViewHeaderFooterView>(cellClass: _Tp.Type,
                                                                            configuration: @escaping (_Tp) -> Void) -> CGFloat {
        return base.fd_heightForHeaderFooterView(withIdentifier: cellClass.reuseIdentifier, configuration: { (headerFooterView) in
            guard let headerFooterView = headerFooterView as? _Tp else {
                fatalError("Could not convert header footer view.")
            }
            configuration(headerFooterView)
        })
    }
}

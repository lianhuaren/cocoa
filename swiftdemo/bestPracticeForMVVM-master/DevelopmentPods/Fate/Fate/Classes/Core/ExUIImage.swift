//
//  ExUIImage.swift
//  Fate
//
//  Created by Archer on 2018/11/23.
//

import FDNamespacer

public extension FOLDin where Base: UIImage {
    
    /// 类似Android中的setColorFilter，可以给图片换颜色。
    /// e.g, UI给了一张绿色的图，实际效果却是红色，这时候要么
    /// 重新换一张红色的图，要么就可以使用这个方法。
    ///
    /// - Parameters:
    ///   - rgbValue: 颜色16进制值
    ///   - alpha: 颜色透明度
    /// - Returns: 滤镜后的图片
    public func setColorFilter(_ rgbValue: Int, _ alpha: CGFloat = 1.0) -> UIImage? {
        return setColorFilter(UIColor(rgbValue: rgbValue, alpha: alpha))
    }
    
    public func setColorFilter(_ fillColor: UIColor) -> UIImage? {
        guard let cgImage = base.cgImage else { return nil }
        UIGraphicsBeginImageContextWithOptions(base.size, false, base.scale)
        let ctx = UIGraphicsGetCurrentContext()
        ctx?.scaleBy(x: 1, y: -1)
        ctx?.translateBy(x: 0, y: -base.size.height)
        ctx?.draw(cgImage, in: CGRect(origin: .zero, size: base.size))
        ctx?.setBlendMode(.sourceAtop)
        ctx?.setFillColor(fillColor.cgColor)
        ctx?.fill(CGRect(origin: .zero, size: base.size))
        let image = UIGraphicsGetImageFromCurrentImageContext()
        UIGraphicsEndImageContext()
        return image
    }
    
    /// 压缩大小
    public func jpegData(maxCount: Int = 0) -> Data? {
        var quality = CGFloat(1)
        var jpegData = base.jpegData(compressionQuality: quality)
        while let data = jpegData, maxCount > 0 &&
            data.count > maxCount &&
            quality > 0 {
                quality -= 0.05
                jpegData = base.jpegData(compressionQuality: quality)
        }
        return jpegData
    }
}

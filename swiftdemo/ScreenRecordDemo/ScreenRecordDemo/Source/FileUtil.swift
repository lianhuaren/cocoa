//
//  FileUtil.swift
//  ScreenRecordDemo
//
//  Created by aaa on 2019/9/29.
//  Copyright © 2019 admin. All rights reserved.
//

import Foundation

class ReplayFileUtil
{
    internal class func createReplaysFolder()
    {
        let documentDirectoryPath = NSSearchPathForDirectoriesInDomains(.documentDirectory, .userDomainMask, true).first
        if let documentDirectoryPath = documentDirectoryPath {
            let replayDirectoryPath = documentDirectoryPath.appending("/Replays")
            let fileManager = FileManager.default
            if !fileManager.fileExists(atPath: replayDirectoryPath) {
                do {
                    try fileManager.createDirectory(atPath: replayDirectoryPath, withIntermediateDirectories: false, attributes: nil)
                } catch {
                    print("Error creating Replays folder in documents dir: \(error)")
                }
            }
        }
    }
    
    internal class func filePath(_ fileName: String) -> String
    {
        createReplaysFolder()
        let paths = NSSearchPathForDirectoriesInDomains(.documentDirectory, .userDomainMask, true)
        let documentsDirectory = paths[0] as String
        let filePath : String = "\(documentsDirectory)/Replays/\(fileName).mp4"
        return filePath
    }
    
    internal class func fetchAllReplays() -> Array<URL>
    {
        let documentsDirectory = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first
        let replayDirectoryPath = documentsDirectory?.appendingPathComponent("/Replays")
        let directoryContents = try! FileManager.default.contentsOfDirectory(at: replayDirectoryPath!, includingPropertiesForKeys: nil, options: [])
        return directoryContents
    }
    
}

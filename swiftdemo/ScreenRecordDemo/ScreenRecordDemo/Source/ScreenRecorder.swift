//
//  ScreenRecorder.swift
//  ScreenRecordDemo
//
//  Created by 佰锐 on 2019/9/29.
//  Copyright © 2019 admin. All rights reserved.
//

import Foundation
import ReplayKit
import UIKit

class ScreenRecorder: NSObject
{
    var assertWriter:AVAssetWriter!
    var videoInput:AVAssetWriterInput!
    
    func startRecording(withFileName fileName: String, recordingHandler:@escaping (Error?)-> Void)
    {
        if #available(iOS 11.0, *)
        {
            let fileURL = URL(fileURLWithPath: ReplayFileUtil.filePath(fileName))
            assertWriter = try! AVAssetWriter(outputURL: fileURL, fileType: AVFileType.mp4)
            let videoOutputSettings: Dictionary<String, Any> = [
                AVVideoCodecKey : AVVideoCodecType.h264,
                AVVideoWidthKey : UIScreen.main.bounds.size.width,
                AVVideoHeightKey : UIScreen.main.bounds.size.height
            ];
            
            videoInput = AVAssetWriterInput (mediaType: AVMediaType.video, outputSettings: videoOutputSettings)
            videoInput.expectsMediaDataInRealTime = true
            assertWriter.add(videoInput)
            
            RPScreenRecorder.shared().startCapture(handler: { (sample, bufferType, error) in
                recordingHandler(error)
                
                if CMSampleBufferDataIsReady(sample)
                {
                    if self.assertWriter.status == AVAssetWriter.Status.unknown
                    {
                        self.assertWriter.startWriting()
                        self.assertWriter.startSession(atSourceTime: CMSampleBufferGetPresentationTimeStamp(sample))
                    }
                    
                    if self.assertWriter.status == AVAssetWriter.Status.failed
                    {
                        print("Error occured, status = \(self.assertWriter.status.rawValue), \(self.assertWriter.error!.localizedDescription) \(String(describing: self.assertWriter.error))")
                        return
                    }
                    
                    if (bufferType == .video)
                    {
                        if self.videoInput.isReadyForMoreMediaData
                        {
                            self.videoInput.append(sample)
                        }
                    }
                }
                
            }) { (error) in
                recordingHandler(error)
            }
        }
    }
    
    func stopRecording(handler: @escaping (Error?) -> Void)
    {
        if #available(iOS 11.0, *)
        {
            RPScreenRecorder.shared().stopCapture { (error) in
                handler(error)
                self.assertWriter.finishWriting {
                    print(ReplayFileUtil.fetchAllReplays())
                }
            }
        }
    }
    
}

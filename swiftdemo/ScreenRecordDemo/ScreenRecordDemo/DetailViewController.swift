//
//  DetailViewController.swift
//  ScreenRecordDemo
//
//  Created by 佰锐 on 2019/9/29.
//  Copyright © 2019 admin. All rights reserved.
//

import UIKit

class DetailViewController: UIViewController {

    @IBOutlet weak var detailDescriptionLabel: UILabel!
    let screenRecord = ScreenRecordCoordinator()

    func configureView() {
        // Update the user interface for the detail item.
        if let detail = detailItem {
            if let label = detailDescriptionLabel {
                label.text = detail.description
            }
        }
    }

    override func viewDidLoad() {
        super.viewDidLoad()
        // Do any additional setup after loading the view.
        configureView()
        
        
        
        screenRecord.viewOverlay?.stopButtonColor = UIColor.red
        let randomNumber = arc4random_uniform(9999)
        screenRecord.startRecording(withFileName: "coolScreenRecording\(randomNumber)", recordingHandler: { (error) in
            print("Recording in progress")
        }) { (error) in
            print("Recording Complete")
        }
        
        DispatchQueue.main.asyncAfter(deadline: .now() + 2.5) { [weak self] in
            self?.screenRecord.stopRecording()
        }
    }
    
    deinit {
        print(String.init(format: "\(#function):%s", object_getClassName(self)))
    }

    var detailItem: NSDate? {
        didSet {
            // Update the view.
            configureView()
        }
    }


}


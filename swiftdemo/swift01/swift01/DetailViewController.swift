//
//  DetailViewController.swift
//  swift01
//
//  Created by  on 2020/8/18.
//  Copyright © 2020 admin. All rights reserved.
//

import UIKit

class DetailViewController: UIViewController {

    @IBOutlet weak var detailDescriptionLabel: UILabel!


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
    }

    var detailItem: Dream? {
        didSet {
            // Update the view.
            configureView()
        }
    }


}


//
//  ViewController.swift
//  swiftdemo2
//
//  Created by 佰锐 on 2019/10/28.
//  Copyright © 2019 admin. All rights reserved.
//

import UIKit
import RxSwift
import RxDataSources

class ViewController: UIViewController {
    
    private let viewModel: AudioBoxViewModelType = AudioBoxViewModel()
    
    override func viewDidLoad() {
        super.viewDidLoad()
        
        performBinding()
        beginRefreshing()
    }

    
}

extension ViewController {
    private func performBinding() {
        viewModel.outputs.bannerLoaded.subscribeNext(weak: self) { (self) -> ([MusicSheetInfo]) -> Void in
            return { (resps) in
                print(resps)
            }
        }
    }
}

extension ViewController {
    private func beginRefreshing() {
        viewModel.inputs.beginRefreshing()
    }
}

//extension ViewController {
//        var type: String!
//        let viewModel = RecommendListViewModel.init()
//        var dataSource: RxTableViewSectionedReloadDataSource<GroupedJHSection>!
//
//        lazy var tableView: UITableView = {
//            let tableView = UITableView.init(frame: view.bounds, style: .grouped)
//            tableView.register(RecommendListCell.self, forCellReuseIdentifier: "RecommendListCell")
//            tableView.backgroundColor = UIColor.yellow
//            tableView.separatorStyle = .none
//            //tableView.tableFooterView = UIView.init()
//            tableView.rowHeight = 50
//            return tableView
//        }()
//
//        override func viewDidLoad() {
//            super.viewDidLoad()
//            // Do any additional setup after loading the view.
//    //        let aa = ZY<NotificationCenter>(NotificationCenter.default)
//    //        aa.post(name: NSNotification.Name("userLogin"), object: nil)
//    //        NotificationCenter.default.zy.post(name: NSNotification.Name("userLogin"), object: nil)
//
//            view.backgroundColor = UIColor.white
//            view.addSubview(tableView)
//
//    //        tableView.snp.makeConstraints { (make) in
//    //            make.top.left.bottom.right.equalTo(view).offset(0)
//    //        }
//
//            initUI()
//            bindModel()
//
//
//
//
//    //        self.type = "推荐"
//    //
//    //        self.viewModel.json = "0-20.json"
//    //        self.viewModel.reload(type:self.type ?? "", json: self.viewModel.json ?? "")
//        }
//
//        func initUI() {
//
//
//        }
//
//        func bindModel() {
//            dataSource = RxTableViewSectionedReloadDataSource<GroupedJHSection>(
//                configureCell: { (ds, tableView, index, model) -> UITableViewCell in
//                    let cell = tableView.dequeueReusableCell(withIdentifier: "RecommendListCell", for: index) as! RecommendListCell
//                    cell.reloadData(data: model)
//                    return cell
//            })
//
//            tableView.rx.setDelegate(self)
//
//            viewModel.data
//            .asObservable()
//            .asDriver(onErrorJustReturn: [])
//            .drive(tableView.rx.items(dataSource: dataSource))
//
//            viewModel.refreshEnd.subscribe(onNext: { () in
//
//            })
//
//        }
//
//        override func viewWillAppear(_ animated: Bool) {
//            super.viewWillAppear(animated)
//
//            SwiftyHUD.show(message: "歌曲已存在")
//        }
//}

//extension ViewController : UITableViewDelegate {
//    func tableView(_ tableView: UITableView, heightForRowAt indexPath: IndexPath) -> CGFloat {
//        let items = dataSource[indexPath.section].items
//        let model = items[indexPath.row]
//        return RecommendListCell.getCellHightData(data:model)
//    }
//}


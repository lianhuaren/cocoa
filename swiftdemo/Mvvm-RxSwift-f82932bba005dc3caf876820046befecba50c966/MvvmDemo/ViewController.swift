//
//  ViewController.swift
//  MvvmDemo
//
//  Created by 徐强强 on 2019/1/31.
//  Copyright © 2019年 徐强强. All rights reserved.
//

import UIKit
import RxDataSources
import RxSwift

class ViewController: UIViewController {

    let disposeBag = DisposeBag()
    
    override func viewDidLoad() {
        super.viewDidLoad()
        // Do any additional setup after loading the view, typically from a nib.
        view.addSubview(tableView)
        
        tableView.snp.makeConstraints { [unowned self] (make) in
            make.left.right.bottom.equalTo(self.view)
            make.top.equalTo(88)
        }
        
        self.getMenusInfo()
            .asDriver(onErrorJustReturn: [])
        .drive(tableView.rx.items(dataSource: dataSource))
                    .disposed(by: disposeBag)
        
    }
    
    func getMenusInfo() -> Single<[ViewController.CellSectionModel]> {
            return RxOpenAPIProvider.rx.request(.newList)
                .mapToObject(type: TopModel.self)
    //            .mapToArray(type: GroupModel.self)
                .asSingle()
                .map { result in
                    let groups = result?.stories
                    return self.createSectionModel(groups: groups)
                }
        }
    
    private func createSectionModel(groups: [GroupModel]?) -> [ViewController.CellSectionModel] {
        if let dishGroups = groups, !dishGroups.isEmpty {
            return [ViewController.CellSectionModel(items: dishGroups)]
        }
        return []
    }
    
    private lazy var dataSource: RxTableViewSectionedReloadDataSource<CellSectionModel> = {
        return RxTableViewSectionedReloadDataSource<CellSectionModel>(configureCell: { [weak self](_, tableView, indexPath, item) -> UITableViewCell in
            let cell: LabelButtonCell = tableView.dequeueReusableCell(withIdentifier: "LabelButtonCell") as! LabelButtonCell
            cell.data = (item.name ?? "", "", "删除", "重命名")
            cell.rightButton1.rx.tap
                .subscribe(onNext: { [weak self] (_) in
//                    self?.cellDeleteButtonTap.onNext(indexPath)
                    let vc = MenuEditGroupViewController()
                    self?.navigationController?.pushViewController(vc, animated: true)
                })
                .disposed(by: cell.disposeBag)
            cell.rightButton2.rx.tap
                .subscribe(onNext: { [weak self] (_) in
//                    self?.cellRenameButtonTap.onNext(indexPath)
                    let vc = MenuEditGroupViewController()
                    self?.navigationController?.pushViewController(vc, animated: true)
                })
                .disposed(by: cell.disposeBag)
            return cell
        })
    }()

    private lazy var tableView: UITableView = {
        let tableView = UITableView(frame: .zero, style: .grouped)
        tableView.delegate = self
        tableView.separatorStyle = .singleLine
        tableView.separatorColor = UIColor.gray
        tableView.estimatedSectionFooterHeight = 0
        tableView.estimatedSectionHeaderHeight = 0
        tableView.showsHorizontalScrollIndicator = false
        tableView.showsVerticalScrollIndicator = false
        let headerView = UIView(frame: CGRect(x: 0,
                y: 0,
                width: self.view.frame.size.width,
                height: 16))
        tableView.tableHeaderView = headerView
        tableView.register(LabelButtonCell.self, forCellReuseIdentifier: "LabelButtonCell")
        return tableView
    }()



}

extension ViewController: UITableViewDelegate {
    func tableView(_ tableView: UITableView, heightForRowAt indexPath: IndexPath) -> CGFloat {
        return 50
    }
}

extension ViewController {
    struct CellSectionModel {
        var items: [Item]
    }
}

extension ViewController.CellSectionModel: SectionModelType {
    typealias Item = GroupModel
    init(original: ViewController.CellSectionModel, items: [Item]) {
        self = original
        self.items = items
    }
}



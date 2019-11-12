//
//  MenuSortDishesViewModel.swift
//  takeaway
//
//  Created by 徐强强 on 2018/7/2.
//  Copyright © 2018年 zaihui. All rights reserved.
//

import Foundation
import RxCocoa
import RxSwift



class MenuSubGroupViewModel {
    private let navigator: MenuSubGroupNavigator
    private var groups: [GroupModel]?
    
    let refreshRelay = PublishSubject<Void>()
    
    
    init(navigator: MenuSubGroupNavigator) {
        self.navigator = navigator
        
        refreshRelay.asObservable()
                   .subscribe(onNext: { (Void) in
                        print("222")
                        self.navigator.toMenuEditGroupVC()
                        .saveData
                        .asDriverOnErrorJustComplete()
                   }).disposed(by: disposeBag)
    }
    
    func transform(input: Input) -> Output {
        
       
//            .subscribe(onNext: { (Void) in
//                print("222")
//                self.navigator.toMenuEditGroupVC()
//                .saveData
//                .asDriverOnErrorJustComplete()
//            }).disposed(by: disposeBag)
        
        
        
        let loadingTracker = ActivityIndicator()
        
        let createNewGroup = input.createNewGroup
            .flatMapLatest { _ in
                self.navigator.toMenuEditGroupVC()
                    .saveData
                    .asDriverOnErrorJustComplete()
            }
        
        let renameGroup = input.cellRenameButtonTap
            .flatMapLatest { indexPath in
                self.navigator.toMenuEditGroupVC()
                    .saveData
                    .asDriverOnErrorJustComplete()
            }
        
        let getMenusInfo = Driver.merge(createNewGroup, input.viewDidLoad, renameGroup)
            .flatMapLatest { _ in
                self.getMenusInfo()
                    .trackActivity(loadingTracker)
                    .asDriver(onErrorJustReturn: self.createSectionModel())
            }
        
        let deleteSubGroups = input.cellDeleteButtonTap
            .flatMapLatest { (indexPath) -> Driver<[MenuSubGroupViewController.CellSectionModel]> in
                return self.deleteSubGroups(at: indexPath)
                    .asDriver(onErrorJustReturn: self.createSectionModel())
            }
        
        let dataSource = Driver.merge(getMenusInfo, deleteSubGroups)
        let loading = loadingTracker.asDriver()
        return Output(dataSource: dataSource)
    }
    
    func getMenusInfo() -> Single<[MenuSubGroupViewController.CellSectionModel]> {
        return RxOpenAPIProvider.rx.request(.newList)
            .mapToObject(type: TopModel.self)
//            .mapToArray(type: GroupModel.self)
            .asSingle()
            .map { result in
                self.groups = result?.stories
                return self.createSectionModel()
            }
    }
    
    private func deleteSubGroups(at indexPath: IndexPath) -> Single<[MenuSubGroupViewController.CellSectionModel]> {
        groups?.remove(at: indexPath.row);
        return Single.just(self.createSectionModel())
    }
    
    private func createSectionModel() -> [MenuSubGroupViewController.CellSectionModel] {
        if let dishGroups = self.groups, !dishGroups.isEmpty {
            return [MenuSubGroupViewController.CellSectionModel(items: dishGroups)]
        }
        return []
    }
}

extension MenuSubGroupViewModel {
    struct Input {
        let createNewGroup: Driver<Void>
        let viewDidLoad: Driver<Void>
        let cellDeleteButtonTap: Driver<IndexPath>
        let cellRenameButtonTap: Driver<IndexPath>
    }
    
    struct Output {
        let dataSource: Driver<[MenuSubGroupViewController.CellSectionModel]>
//        let loading: Driver<Bool>
    }
}

fileprivate var disposeBagContext: UInt8 = 0

/// each HasDisposeBag offers a unique RxSwift DisposeBag instance
public protocol HasDisposeBag: class {

    /// a unique RxSwift DisposeBag instance
    var disposeBag: DisposeBag { get set }
}

extension HasDisposeBag {

    func synchronizedBag<T>( _ action: () -> T) -> T {
        objc_sync_enter(self)
        let result = action()
        objc_sync_exit(self)
        return result
    }

    public var disposeBag: DisposeBag {
        get {
            return synchronizedBag {
                if let disposeObject = objc_getAssociatedObject(self, &disposeBagContext) as? DisposeBag {
                    return disposeObject
                }
                let disposeObject = DisposeBag()
                objc_setAssociatedObject(self, &disposeBagContext, disposeObject, .OBJC_ASSOCIATION_RETAIN_NONATOMIC)
                return disposeObject
            }
        }

        set {
            synchronizedBag {
                objc_setAssociatedObject(self, &disposeBagContext, newValue, .OBJC_ASSOCIATION_RETAIN_NONATOMIC)
            }
        }
    }
}



extension MenuSubGroupViewModel: HasDisposeBag {}

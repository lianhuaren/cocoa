/*
*  This file is part of FFL project.
*
*  The MIT License (MIT)
*  Copyright (C) 2017-2018 zhufeifei All rights reserved.
*
*  FFL_NetFdEvent.cpp
*  Created by zhufeifei(34008081@qq.com) on 2018/12/1
*  https://github.com/zhenfei2016/FFLv2-lib.git
*  网络读写的时间驱动类
*
*/
#include <net/FFL_NetEventLoop.hpp>
#include <net/FFL_Net.h>
#include "FFL_SocketPair.hpp"
#include "internalLogConfig.h"
#include <list>

namespace FFL {
	const static int  FD_LIST_MAX_SIZE = 64;

	//const static int8_t READABLE = 0x01;
	//const static int8_t WRITABLE = 0x10;

    #define List std::list
	class EventPacket {
	public:
		int64_t mEventIndex;
		//
		//  添加还是删除指令
		//
		bool mCommandAdd;
		bool mCommandRemove;
		//
		//  传输的参数
		//
		NetFD fd;
		NetEventLoop::Callback* readHandler;
		NetEventLoop::CallbackFree readHandlerFree;
		void* priv;
	};

	class NetEventLoopImpl {
	public:
		NetEventLoopImpl(NetEventLoop* eventloop,int64_t evenloopWaitUs);
		~NetEventLoopImpl();

		bool addFd(NetFD fd,
			NetEventLoop::Callback* readHandler,
			NetEventLoop::CallbackFree readHandlerFree,
			void* priv);
		bool removeFd(NetFD fd);
		bool onStart();
		void onStop();		
		bool eventLoop(int32_t* waitTime);


	protected:
		NetEventLoop* mEventLoop;

		struct  FdEntry {
			//
			// 句柄
			//
			NetFD mFd;
			//
			//  移除这个fd
			//
			bool mRemoved;
			//
			//  当前状态，可读 0x01  | 可写 0x010
			//
			int8_t mFlag;
			//
			//  读写处理handler
			//
			NetEventLoop::Callback* mReadHandler;
			NetEventLoop::CallbackFree mReadHandlerFree;
			void* mPriv;

		};
		FdEntry* findFdEntry(NetFD fd);

		bool processAddFd(NetFD fd,
			NetEventLoop::Callback* readHandler,
			NetEventLoop::CallbackFree readHandlerFree,
			void* priv);
		//
		//  移除这个句柄的处理handler
		//
		bool processRemoveFd(FdEntry* entry);
		//
		//  处理一下可读的fd,返回是否技术
		//
		bool processReadableFd(NetFD* fdList, int8_t* readableFlagList, int32_t numFd);
	private:
		//
		//  本系统的控制端口
		//
		FFL::SocketPair* mSocketPairControl;
		NetFD mControlFd;
		bool mOnlyTryControlFd;
		int64_t mEventNextId;
		bool processControlEvent(NetFD fd, bool readable, bool writeable, bool exception, void* priv);
	private:
		//
		//  停止的标志
		//
		volatile bool mStopLoop;
		//
		//  轮训等待时长
		//
		int64_t  mWaitUs;
		//
		//  管理的fd列表
		//
		FdEntry* mFdList;
		int mFdNum;

	private:
		void addEvent(EventPacket* event);
		void removeEvent(EventPacket* event);
		//
		//  当前所有的add,remove事件
		//
		FFL::CMutex mEventsLock;

		
		List<EventPacket*> mPendingEvents;
	};
	NetEventLoopImpl::NetEventLoopImpl(NetEventLoop* eventloop,int64_t evenloopWaitUs):
		mSocketPairControl(NULL),
        mControlFd(0),
        mOnlyTryControlFd(false),
	    mStopLoop(false),
        mWaitUs(evenloopWaitUs),
		mFdNum(0){

		mEventLoop=eventloop;
		mEventNextId = 0;
		mFdList = new FdEntry[FD_LIST_MAX_SIZE];
		memset(mFdList, 0,sizeof(mFdList[0])*FD_LIST_MAX_SIZE);
	}

	NetEventLoopImpl::~NetEventLoopImpl() {
		FFL_SafeFreeA(mFdList);
		FFL_SafeFree(mSocketPairControl);
		if (mControlFd) {
			FFL_socketClose(mControlFd);
			mControlFd = 0;
		}
	}

	//
	// 添加一个监听的句柄， readHandler如果不等NULL，则在removeFd的时候会调用 readHandlerFree进行清理
	// 
	bool NetEventLoopImpl::addFd(NetFD fd,
		NetEventLoop::Callback* readHandler,
		NetEventLoop::CallbackFree readHandlerFree,
		void* priv) {
		if (fd == INVALID_NetFD || readHandler ==NULL ) {
			return false;
		}	

		if (mControlFd == INVALID_NetFD) {
			//
			// 还没有启动的情况下
			//
			return  processAddFd(fd, readHandler, readHandlerFree, priv);

		}

		EventPacket* packet = new EventPacket();
		packet->mEventIndex=mEventNextId++;
		packet->mCommandAdd = true;
		packet->mCommandRemove = false;
		packet->fd = fd;
		packet->readHandler = readHandler;
		packet->readHandlerFree = readHandlerFree;
		packet->priv = priv;
		if (!mSocketPairControl->writeFd0((uint8_t*)(&packet), sizeof(packet), 0)) {
			FFL_SafeFree(packet);
			return false;
		}

		addEvent(packet);
		return true;
	}
	bool NetEventLoopImpl::removeFd(NetFD fd) {
		if (fd == INVALID_NetFD) {
			return false;
		}
		if (mControlFd == INVALID_NetFD) {
			NetEventLoopImpl::FdEntry* entry = findFdEntry(fd);
			return processRemoveFd(entry);
		}

		EventPacket* packet = new EventPacket();
		packet->mEventIndex = mEventNextId++;
		packet->mCommandAdd = false;
		packet->mCommandRemove = true;
		packet->fd = fd;
		packet->readHandler = NULL;
		packet->readHandlerFree = NULL;
		packet->priv = NULL;
		if (!mSocketPairControl->writeFd0((uint8_t*)(&packet), sizeof(packet), 0)) {
			FFL_SafeFree(packet);
			return false;
		}
		
		addEvent(packet);
		return true;
	}
	//
	//   如果start使用了EventloopThread，则stop会阻塞到线程退出
	//   否则则仅仅置一下标志
	//
	bool NetEventLoopImpl::onStart() {
		mStopLoop = false;
		FFL_SafeFree(mSocketPairControl);
		mSocketPairControl = new SocketPair();
		if (!mSocketPairControl->create()) {
			FFL_SafeFree(mSocketPairControl);
			return false;
		}	

		mControlFd = mSocketPairControl->getFd1();
		return true;
	}
	void NetEventLoopImpl::onStop() {
		//
		//  触发退出select
		//
		mStopLoop = true;
		if (mSocketPairControl) {
			uint8_t c = 'x';
			mSocketPairControl->writeFd0(&c, 1,0);
			mSocketPairControl->destroy();
		}	
	}
	//
	//   阻塞的线程中执行的eventloop,返回是否继续进行eventLoop
	//   waitTime:输出参数，下一次执行eventLoop等待的时长
	//   true  : 继续进行下一次的eventLoop
	//   false : 不需要继续执行eventloop
	//
	bool NetEventLoopImpl::eventLoop(int32_t* waitTime){
		if (!mEventLoop->isStarted()) {
			INTERNAL_FFL_LOG_WARNING("NetEventLoop: Failed to NetEventLoop::eventLoop. not start.");
			return false;
		}		

		
		NetFD fdList[FD_LIST_MAX_SIZE] = {};
		int numFd = 1;
		fdList[0] = mControlFd;

		//
		//  外部add的
		//
		if (!mOnlyTryControlFd) {
			for (int i = 0; i < FD_LIST_MAX_SIZE && numFd <= mFdNum; i++) {
				if (mFdList[i].mRemoved || mFdList[i].mFd == INVALID_NetFD) {
					continue;
				}
				fdList[numFd] = mFdList[i].mFd;
				numFd += 1;
			}
		}
				
		INTERNAL_FFL_LOG_DEBUG("NetEventLoop: select fdNum=%d mOnlyTryControlFd=%d",numFd, mOnlyTryControlFd?1:0);
		int8_t flagList[FD_LIST_MAX_SIZE] = {};
		int32_t selectRet=FFL_socketSelect(fdList, flagList, numFd, mWaitUs);
		if (selectRet > 0) {
			//
			// 派发可读消息
			//
			if (!processReadableFd(fdList, flagList, numFd)) {
				return false;
			}
			mOnlyTryControlFd = false;
		}else  if (selectRet == 0) {
			INTERNAL_FFL_LOG_DEBUG("NetEventLoop: select timeout waitUs=%" lld64, mWaitUs);
		}else {
			//
			//  不使用锁，没问题
			//
			if (mStopLoop) {
				return false;
			}

			{   //
				//  处理一下可能没有发送过来的消息
				//
				CMutex::Autolock l(mEventsLock);
				for (List<EventPacket*>::iterator it = mPendingEvents.begin(); it != mPendingEvents.end(); ) {
					EventPacket* packet = (*it);
					if (packet) {
						if (packet->mCommandAdd) {
							processAddFd(packet->fd, packet->readHandler, packet->readHandlerFree, packet->priv);
						}
						if (packet->mCommandRemove) {
							NetEventLoopImpl::FdEntry* entry = findFdEntry(packet->fd);
							processRemoveFd(entry);
						}
					}
					it = mPendingEvents.erase(it);
				}
			}

			INTERNAL_FFL_LOG_DEBUG("NetEventLoop: select failed");			
			mOnlyTryControlFd = false;
			FFL_sleep(100);
		}	

		return true;
	}


	bool NetEventLoopImpl::processAddFd(NetFD fd,
		NetEventLoop::Callback* readHandler,
		NetEventLoop::CallbackFree readHandlerFree,
		void* priv) {

		for (int32_t i = 0; i < FD_LIST_MAX_SIZE; i++) {
			FdEntry* entry = mFdList + i;
			if (fd == entry->mFd) {
				if (entry->mRemoved) {
					entry->mFd = 0;
					mFdNum--;
					if (entry->mReadHandlerFree) {
						entry->mReadHandlerFree(entry->mReadHandler);
					}
				}
				else {
					return false;
				}
			}
			if (entry->mFd != INVALID_NetFD) {
				continue;
			}

			entry->mFd = fd;
			entry->mReadHandler = readHandler;
			entry->mReadHandlerFree = readHandlerFree;
			entry->mPriv = priv;
			entry->mRemoved = false;
			mFdNum++;
			INTERNAL_FFL_LOG_DEBUG("NetEventLoop: add fdNum=%d",mFdNum+1);
			break;
		}
		return true;
	}
	//
	//  移除这个句柄的处理handler
	//
	bool NetEventLoopImpl::processRemoveFd(NetEventLoopImpl::FdEntry* entry) {
	    bool ret=false;
		if (entry) {
            if (entry->mReadHandlerFree) {
                entry->mReadHandlerFree(entry->mReadHandler);
            }
            mFdNum--;
            INTERNAL_FFL_LOG_INFO("NetEventLoopImpl: removefd=%d fdListNum=%d", entry->mFd,
                                  mFdNum + 1);
            memset(entry, 0, sizeof(NetEventLoopImpl::FdEntry));
            ret=true;
        }
		return ret;
	}

	//
	//  处理一下可读的fd
	//
	bool NetEventLoopImpl::processReadableFd(NetFD* fdList, int8_t* readableFlagList,int32_t numFd) {
		if (readableFlagList[0]) {
			//
			//  本系统控制端口
			//
			return processControlEvent(mControlFd, true, false, false,  0);			
		}
		
		for (int32_t i = 1; i < numFd; i++) {
			if (readableFlagList[i]==0) {
				continue;
			}

			NetEventLoopImpl::FdEntry* entry = findFdEntry(fdList[i]);
			if (entry == NULL) {
				continue;
			}

			if (!entry->mRemoved) {
				//
				//  可以读了
				//				
				INTERNAL_FFL_LOG_DEBUG("NetEventLoopImpl: onNetEvent  fd=%d", entry->mFd);
				if (!entry->mReadHandler->onNetEvent(entry->mFd, true, false, false, entry->mPriv)) {
					//
					//  返回false，则不需要了
					//
					entry->mRemoved = true;
				}
			}

			if (entry->mRemoved) {
				INTERNAL_FFL_LOG_DEBUG("NetEventLoopImpl: onNetEvent return false. fd=%d", entry->mFd);
				processRemoveFd(entry);
			}
		}

		return true;
	}
	
	NetEventLoopImpl::FdEntry* NetEventLoopImpl::findFdEntry(NetFD fd) {
		for (int32_t i = 0; i < FD_LIST_MAX_SIZE; i++) {
			if (mFdList[i].mFd == fd) {
				return mFdList + i;
			}
		}
		return NULL;
	}


	//
	//  返回是否还继续读写
	//
	bool NetEventLoopImpl::processControlEvent(NetFD fd, bool readable, bool writeable, bool exception,void* priv) {
		if (!readable) {
			return true;
		}

		EventPacket* packet = NULL;
		size_t readed = 0;
		if (FFL_OK != FFL_socketRead(fd, &packet, sizeof(packet), &readed)) {
			readed = 1;
		}
		if (readed == 1) {	
			//
			//  退出系统
			//
			INTERNAL_FFL_LOG_DEBUG("NetEventLoopImpl: processControlEvent quit");
			return false;
		}
		if (readed != sizeof(packet) || packet ==NULL ) {
			return true;
		}

        NetEventLoopImpl::FdEntry* entry=NULL;
		if (packet->mCommandAdd) {		
			processAddFd(packet->fd, packet->readHandler, packet->readHandlerFree, packet->priv);
		}else if (packet->mCommandRemove) {
			entry = findFdEntry(packet->fd);
			processRemoveFd(entry);
		}

		removeEvent(packet);
		FFL_SafeFree(packet);
		return true;
	}

	void NetEventLoopImpl::addEvent(EventPacket* event) {
		CMutex::Autolock l(mEventsLock);
		mPendingEvents.push_back(event);
	}
	void NetEventLoopImpl::removeEvent(EventPacket* event) {
		CMutex::Autolock l(mEventsLock);
		for (List<EventPacket*>::iterator it = mPendingEvents.begin(); it != mPendingEvents.end(); ) {
			if (*it == event) {
				it=mPendingEvents.erase(it);
			}
			else {
				it++;
			}
		}
	}

	//////////////////////////////////////////////////////////////////////////////////////////////
	//
	//  evenloopWaitUs:句柄多长时间轮训一次，默认0，一直轮训
	//   <0 ,一直等待到有数据
	//   >0  等待evenloopWaitUs毫秒
	//
	NetEventLoop::NetEventLoop(int64_t evenloopWaitUs ) {
		mImpl = new NetEventLoopImpl(this,evenloopWaitUs);
	}
	NetEventLoop::~NetEventLoop() {
		FFL_SafeFree(mImpl);
	}
	//
	// 添加一个监听的句柄， readHandler如果不等NULL，则在removeFd的时候会调用 readHandlerFree进行清理
	// priv :透传到fdReady中
	// 
	bool NetEventLoop::addFd(NetFD fd,
		NetEventLoop::Callback* readHandler,
		NetEventLoop::CallbackFree readHandlerFree,
		void* priv) {
		return mImpl->addFd(fd, readHandler, readHandlerFree, priv);
	}
	//
	//  移除这个句柄的处理handler
	//
	bool NetEventLoop::removeFd(NetFD fd) {
		return mImpl->removeFd(fd);
	}
	//
	//  调用。start，stop会触发onStart,onStop的执行
	//  onStart :表示准备开始了 ,可以做一些初始化工作
	//  onStop :表示准备停止了 ,可以做停止前的准备，想置一下信号让eventloop别卡住啊 
	//  在这些函数中，不要再调用自己的函数，例如：start,stop, isStarted等
	//
	bool NetEventLoop::onStart() {
		return mImpl->onStart();
	}
	void NetEventLoop::onStop() {
		return mImpl->onStop();
	}
	//
	//   阻塞的线程中执行的eventloop,返回是否继续进行eventLoop
	//   waitTime:输出参数，下一次执行eventLoop等待的时长
	//   true  : 继续进行下一次的eventLoop
	//   false : 不需要继续执行eventloop
	//
	bool NetEventLoop::eventLoop(int32_t* waitTime) {
		return mImpl->eventLoop(waitTime);
	}
}

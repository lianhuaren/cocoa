#include <FFL_BlockList.hpp>

namespace FFL {

	template < class T >
	class BlockingListImpl{
	public:
		BlockingListImpl::BlockingListImpl(const char* name) :mIsStarted(false), mMaxSize(0) {
			mName = name ? name : "";
		}
		//
		//  name：list的名称，
		//  maxSize : 最多保存几个元素
		//
		BlockingListImpl(const char* name, uint32_t maxSize) :mIsStarted(false), mMaxSize(maxSize) {
			mName = name ? name : "";
		}
		~BlockingListImpl() {
			clear();
			stop();
		}
	public:
		//
		//  启动，停止多线程的list
		//
		void start() {
			FFL::CMutex::Autolock l(mMutex);
			mIsStarted = true;
		}
		void stop() {
			FFL::CMutex::Autolock l(mMutex);
			if (mIsStarted) {
				mIsStarted = false;
				mCond.signal();
			}
		}
		//
		//  插入一条数据
		//
		bool incoming(T& element) {
			FFL::CMutex::Autolock l(mMutex);
			if (!mIsStarted) {
				FFL_LOG_WARNING("BlockingList(%s) not start. incoming",mName.string());
				return false;
			}

			//
			//  等待缓冲区少一点
			//
			size_t size = 0;
			while (mMaxSize > 0 && mIsStarted) {
				size = mDataList.size();
				if (size >= mMaxSize) {
					FFL_LOG_DEBUG("BlockingList(%s) full. incoming element maxSize=%u ,size=%u  ",
						mName.string(), mMaxSize, mDataList.size());
					mCond.wait(mMutex);
				}
			}
			if (!mIsStarted) {
				return false;
			}

			//
			//  添加数据
			//
			bool sign = (size == 0);
			mDataList.push_back(element);
			if (sign) {
				mCond.signal();
			}
			return true;
		}

		//
		//  获取最先插入的数据，没有数据则等待，一直到插入或者stop
		//  errNo：如果设置了就返回错误码，正常为0
		//
		T next(int32_t* errNo) {
			T elm;
			FFL::CMutex::Autolock l(mMutex);
			while (true && mIsStarted) {
				if (mDataList.size() > 0) {
					elm = mDataList.front();
					mDataList.pop_front();
					return elm;
				}

				FFL_LOG_DEBUG("BlockingList(%s) is empty.", mName.string());
				mCond.wait(mMutex);
			}

			if (*errNo) {
				*errNo = -1;
			}
			return elm;
		}

		//
		//   清空数据list
		//
		void clear() {
			FFL::CMutex::Autolock l(mMutex);
			mDataList.clear();
		}
		//
		//   清空数据list
		//
		void clear(std::list< T >& tmp) {
			FFL::CMutex::Autolock l(mMutex);
			for (std::list< T >::iterator it = mDataList.begin(); it1 = mDataList.end(); it++) {
				tmp.push_back(*it);
			}			  
			mDataList.clear();
		}

		//
		//  获取数据大小
		//
		uint32_t getSize() {
			FFL::CMutex::Autolock l(mMutex);
			return mDataList.size();
		}
	private:
		String mName;

		FFL::CMutex mMutex;
		volatile bool mIsStarted;
		uint32_t mMaxSize;

		FFL::CCondition mCond;
		std::list< T > mDataList;
	};
}
#endif
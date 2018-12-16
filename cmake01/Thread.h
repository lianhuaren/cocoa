#ifndef THREAD_H
#define THREAD_H

#include <pthread.h>
#include <functional>
#include <iostream>
#include <assert.h>
#include <string>


class thread_data
{
	public:
		thread_data(std::function<void()> &f):fun(f){}
		std::function<void()> fun;
};

class Thread
{
	public:
		explicit	Thread(std::function<void()> threadFunction, std::string thread_name):thread_function_(threadFunction), name_(thread_name)
	{
		start_ =false;
		join_ = false;
	}

		~Thread()
		{
			if(start_ && !join_)
			{
				pthread_detach(tid_);
			}
		}

		std::string name()
		{
			return name_;
		}

		void start()
		{
			assert(!start_);
			start_ = true;
			thread_data *thread_data_ptr = new thread_data(thread_function_);
			pthread_create(&tid_, NULL, run, thread_data_ptr);
		}

		void stop()
		{
			pthread_detach(tid_);
		}

		int join()
		{
			assert(!join_);
			join_ = true;
			return pthread_join(tid_, NULL);
		}

		Thread(const Thread&) = delete;
		Thread& operator=(const Thread&) = delete;

	private:

		static void* run(void* obj)
		{
			//±ÿ–Î «æ≤Ã¨µƒ≥…‘±∫Ø ˝≤≈ø…“‘±ªµ˜”√
			//ƒ«√¥Œ Ã‚¿¥¡À ‘ı√¥∞Ï
			thread_data* data = static_cast<thread_data*>(obj);
			data->fun();
			delete data;
			return NULL;
		}

		std::function<void()> thread_function_;
		pthread_t tid_;
		bool start_;
		bool join_;
		std::string name_;
};

#endif

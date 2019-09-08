#ifndef THREAD_POOL_H
#define THREAD_POOL_H

#include <iostream>
#include <thread>
#include <vector>
#include <functional>
#include <queue>
#include <condition_variable>
#include <mutex>
#include <atomic>

namespace dadt {

// a simple thread pool
class ThreadPool {
private:
  // thread
  std::vector<std::thread> workers_;

  // a queue
  std::queue<std::function<void()>> task_queue_;

  // whether it has been stopped
  std::atomic<bool> stopped_;

  //the mutex
  std::mutex mutex_;

  //the condition variable
  std::condition_variable cond_var_;

public:
  ThreadPool();

  void init(int thread_count = 1);

  // put a task in thread pool
  void enqueue(std::function<void()> &&task);

  // stop the thread pool
  void stop();

  // do the task
  void do_task();
};

}

#endif
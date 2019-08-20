#ifndef THREAD_POOL_H
#define THREAD_POOL_H

#include <iostream>
#include <thread>
#include <vector>

#include "concurrentqueue.h"

namespace dadt {

// a simple thread pool
class ThreadPool {
private:
  // thread
  std::vector<std::thread> threads_;

  // a task queue
  moodycamel::ConcurrentQueue<std::function<void()>> queue_;

  // if it has been stopped
  std::atomic<bool> stopped_;

public:
  ThreadPool(int thread_count = 2);

  // put a task in thread pool
  void enqueue(std::function<void()> &&task);

  // stop the thread pool
  void stop();

  void do_task();
};

}

#endif
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

#include "context.h"
#include "concurrentqueue.h"

namespace dadt {

// a Barrier: contain a atomic counter will increase 1 when task in pool
// than when finish decrease 1
class Barrier {
private:
  std::atomic<int64_t> counter_;

  std::mutex mutex_;

  std::condition_variable cv_;

public:
  Barrier();

  // increase counter_
  void increase();

  // decrease counter_
  void decrease();

  // same decrease
  void notify();

  // wait counter_ become to 0
  void wait();
};

// a simple thread pool
class ThreadPool {
private:
  // barrier
  Barrier barrier_;

  // thread pool
  std::vector<std::thread> workers_;

  // whether it has been stopped
  std::atomic<bool> stopped_;

  // a lock-free queue
  moodycamel::ConcurrentQueue<std::function<void()>> task_queue_;

  //the mutex
  std::mutex mutex_;

  //the condition variable
  std::condition_variable cond_var_;

public:
  ThreadPool();

  // initialize
  void init(int thread_count);

  // put a task in thread pool
  void enqueue(std::function<void()> &&task);

  void wait();

  // stop the thread pool
  void stop();

  // do the task
  void do_task();
};

}

#endif
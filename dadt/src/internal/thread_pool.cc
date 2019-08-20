#include "thread_pool.h"

namespace dadt {

ThreadPool::ThreadPool(int thread_count): stopped_(false) {
  for (int i = 0; i < thread_count; ++i) {
    std::thread thread(&ThreadPool::do_task, this);

    threads_.emplace_back(std::move(thread));
  }
}

// put a task in thread pool
void ThreadPool::enqueue(std::function<void()> &&task) {
  queue_.enqueue(task);
}

// stop the thread pool
void ThreadPool::stop() {
  stopped_ = true;

  for (auto &t : threads_) {
    t.join();
  }
}

void ThreadPool::do_task() {
  while (true) {
    std::function<void()> task;
    
    if (queue_.try_dequeue(task)) {
      task();
    } else if (stopped_) {
      break;
    }
  }
}

}
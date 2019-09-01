#include "definition.h"
#include "thread_pool.h"

namespace dadt {

ThreadPool::ThreadPool(): stopped_(false) {
}

void ThreadPool::init(int thread_count) {
  for (int i = 0; i < thread_count; ++i) {
    std::thread thread(&ThreadPool::do_task, this);

    workers_.emplace_back(std::move(thread));
  }
}

// put a task in thread pool
void ThreadPool::enqueue(std::function<void()> &&task) {
  ARGUMENT_CHECK(false == stopped_, "the thread pool has been stopped");
  
  {
    // try to get mutex
    std::unique_lock<std::mutex> lock(mutex_);

    // add to queue
    task_queue_.emplace(task);
  }

  cond_var_.notify_one();
}

// stop the thread pool
void ThreadPool::stop() {
  stopped_ = true;

  cond_var_.notify_all();

  for (auto &t : workers_) {
    t.join();
  }
}

void ThreadPool::do_task() {
  while (true) {
    // try to get mutex
    std::unique_lock<std::mutex> lock(mutex_);

    // wait
    cond_var_.wait(lock, [this] {
      return this->stopped_ || !this->task_queue_.empty();
    });

    // if the queue has task
    if (!task_queue_.empty()) {
      auto task = std::move(task_queue_.front());
      task_queue_.pop();

      // unlock the mutex
      lock.unlock();

      // run task
      task();
    } else if (this->stopped_) {
      break;
    }
  }
}

}
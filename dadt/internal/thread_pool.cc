#include "definition.h"
#include "thread_pool.h"

namespace dadt {

Barrier::Barrier(): counter_(0) {
}

void Barrier::increase() {
  counter_++;
}

void Barrier::decrease() {
  notify();
}

void Barrier::notify() {
  auto after_val = counter_.fetch_sub(1) - 1;

  if (0 != after_val) {
    ARGUMENT_CHECK(after_val >= 0, "after_val is:" << after_val << ", but must >= 0");

    return;
  }

  std::unique_lock<std::mutex> lock(mutex_);
  cv_.notify_all();
}

void Barrier::wait() {
  std::unique_lock<std::mutex> lock(mutex_);

  while (0 != counter_.load()) {
    cv_.wait(lock);
  }
}

ThreadPool::ThreadPool(): stopped_(false) {
}

void ThreadPool::init(int thread_count) {
  for (int i = 0; i < thread_count; ++i) {
    std::thread thread(&ThreadPool::do_task, this);

    workers_.emplace_back(std::move(thread));
  }
}

void ThreadPool::enqueue(std::function<void()> &&task) {
  ARGUMENT_CHECK(false == stopped_.load(), "the thread pool has been stopped");

  // increase barrier
  barrier_.increase();

  // put task in queue
  ARGUMENT_CHECK(task_queue_.enqueue(task), "enqueue task to threadpool get error!");

  // notify one thread
  std::unique_lock<std::mutex> lock(mutex_);
  cond_var_.notify_one();
}

void ThreadPool::wait() {
  barrier_.wait();
}

void ThreadPool::stop() {
  stopped_ = true;

  cond_var_.notify_all();

  for (auto &t : workers_) {
    t.join();
  }
}

void ThreadPool::do_task() {
  while (true) {
    // try to get a task
    std::function<void()> task;

    // task_queue_ is a lock-free queue, so no need get the mutex.
    if (task_queue_.try_dequeue(task)) {
      // run task
      task();

      // notify barrier
      barrier_.notify();
    } else if (this->stopped_.load()) {
      // if has been stopped just break;
      break;
    } else {
      // when not get a task and not stopped, need to sleep.
      std::unique_lock<std::mutex> lock(mutex_);

      // wait
      cond_var_.wait(lock, [this] {
        return this->stopped_ || (0 != this->task_queue_.size_approx());
      });
    }
  }
}

}
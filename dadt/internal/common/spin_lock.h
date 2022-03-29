#pragma once

#include <atomic>
#include <iostream>

namespace dadt {

// ref:https://rigtorp.se/spinlock/
class SpinLock {
private:
  std::atomic<bool> lock_ = {false};

public:
  SpinLock();

  SpinLock(const SpinLock&) = delete;
  SpinLock(SpinLock&&) = delete;

  ~SpinLock() = default;

public:
  void lock();

  void unlock();
};

// Simple Handler
class SpinLockHandler {
private:
  SpinLock& spin_lock_;

public:
  explicit SpinLockHandler(SpinLock& spin_lock);

  ~SpinLockHandler();
};

// use atomic to implement a simple spin status lock
class SpinStatusLock {
  friend class LockTensor;

private:
  // use a atomic
  std::atomic<int32_t> lock_;

private:
  SpinStatusLock(int32_t init_value);

  // wait until the lock is old_value
  // than chang it to new_value
  void CompareExchange(int32_t old_value, int32_t new_value);

  // if the lock is new_value than return
  // if the lock is old_value than change it to new_value return
  // if the lock is another value just wait
  void Wait(int32_t old_value, int32_t new_value);
};

}  // namespace dadt

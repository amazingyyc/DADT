#include "common/spin_lock.h"

namespace dadt {

SpinLock::SpinLock() {
}

void SpinLock::lock() {
  for (;;) {
    if (!lock_.exchange(true, std::memory_order_acquire)) {
      break;
    }

    while (lock_.load(std::memory_order_relaxed)) {
      __builtin_ia32_pause();
    }
  }
}

void SpinLock::unlock() {
  lock_.store(false, std::memory_order_release);
}

SpinLockHandler::SpinLockHandler(SpinLock& spin_lock) : spin_lock_(spin_lock) {
  spin_lock_.lock();
}

SpinLockHandler::~SpinLockHandler() {
  spin_lock_.unlock();
}

SpinStatusLock::SpinStatusLock(int32_t init_value) : lock_(init_value) {
}

void SpinStatusLock::CompareExchange(int32_t old_value, int32_t new_value) {
  auto expected = old_value;

  while (!lock_.compare_exchange_weak(expected, new_value)) {
    expected = old_value;
  }
}

void SpinStatusLock::Wait(int32_t old_value, int32_t new_value) {
  auto expected = new_value;

  while (true) {
    expected = new_value;

    if (lock_.compare_exchange_weak(expected, new_value)) {
      break;
    }

    expected = old_value;

    if (lock_.compare_exchange_weak(expected, new_value)) {
      break;
    }
  }
}

}  // namespace dadt
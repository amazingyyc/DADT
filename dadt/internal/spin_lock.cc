#include "spin_lock.h"

namespace dadt {

SpinStatusLock::SpinStatusLock(int initialize_value): lock_(initialize_value) {
}

void SpinStatusLock::compare_exchange(int old_value, int new_value) {
  auto expected = old_value;

  while (!lock_.compare_exchange_weak(expected, new_value)) {
    expected = old_value;
  }
}

void SpinStatusLock::wait(int old_value, int new_value) {
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


SpinLock::SpinLock() {
  lock_.clear();
}

void SpinLock::lock() {
  while (lock_.test_and_set(std::memory_order_acquire));
}

void SpinLock::unlock() {
  lock_.clear(std::memory_order_release);
}


SpinLockHandler::SpinLockHandler(SpinLock &spin_lock): spin_lock_(spin_lock) {
  spin_lock_.lock();
}

SpinLockHandler::~SpinLockHandler() {
  spin_lock_.unlock();
}

}
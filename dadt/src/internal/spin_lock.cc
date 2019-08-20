#include "spin_lock.h"

namespace dadt {

SpinLock::SpinLock(int initialize_value): lock_(initialize_value) {
}

/**
 * wait until the lock is old_value
 * than chang it to new_value
 */
void SpinLock::compare_exchange(int old_value, int new_value) {
  auto expected = old_value;

  while (!lock_.compare_exchange_weak(expected, new_value)) {
    expected = old_value;
  }
}

/**
 * if the lock is new_value than return
 * if the lock is old_value than change it to new_value return
 * if the lock is another value just wait
 */
void SpinLock::wait(int old_value, int new_value) {
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

}
#include "spin_lock.h"

namespace dadt {

SpinLock::SpinLock(int initialize_value): lock_(initialize_value) {
}

void SpinLock::compare_exchange(int old_value, int new_value) {
  auto expected = old_value;

  while (!lock_.compare_exchange_weak(expected, new_value)) {
    expected = old_value;
  }
}

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
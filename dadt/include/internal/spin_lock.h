#ifndef SPIN_LOCK
#define SPIN_LOCK

#include <iostream>
#include <atomic>

namespace dadt {

/**use atomic to implement a simple spin lock*/
class SpinLock {
private:
  // use a atomic
  std::atomic<int> lock_;

public:
  SpinLock(int initialize_value);

  /**
   * wait until the lock is old_value
   * than chang it to new_value
   */
  void compare_exchange(int old_value, int new_value);

  /**
   * if the lock is new_value than return
   * if the lock is old_value than change it to new_value return
   * if the lock is another value just wait
   */
  void wait(int old_value, int new_value);
};

}

#endif
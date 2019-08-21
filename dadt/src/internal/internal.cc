#include "exception.h"
#include "internal.h"
#include "commander.h"

namespace dadt {

// the commander
Commander commander_;

// initialize dadt
void init() {
  commander_.init();
}

// id have been initialized
bool initialized() {
  return commander_.initialized();
}

// how many process 
int size() {
  return commander_.size();
}

// how many process in current machine 
int local_size() {
  return commander_.local_size();
}

// the rank of current process
int rank() {
  return commander_.rank();
}

// local rank
int local_rank() {
  return commander_.local_rank();
}

// barrier all process
void barrier() {
  commander_.barrier();
}

// local barrier all process
void local_barrier() {
  commander_.local_barrier();
}

// get a interim tensor by TaskType
std::shared_ptr<LockTensor> get_interim_tensor(TaskType task_type, 
                                              std::string name, 
                                              std::vector<int> dims, 
                                              ElementType element_type) {
  return commander_.get_interim_tensor(task_type, name, dims, element_type);
}

// put a task is async queue
void async_task(std::function<void()> &&task) {
  commander_.async_task(std::move(task));
}

}
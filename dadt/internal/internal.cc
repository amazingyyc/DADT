#include "exception.h"
#include "internal.h"
#include "commander.h"

namespace dadt {

// the commander
dadt::Commander commander_;

// initialize dadt
void init() {
  commander_.init();
}

// stop the background thread
void shutdown() {
  commander_.shutdown();
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

// put a task in queue
void enqueue_task(Task &&t) {
  commander_.enqueue_task(std::move(t));
}

// put a task is async queue
void async_job(std::function<void()> &&job) {
  commander_.async_job(std::move(job));
}

std::shared_ptr<LockTensor> have_midway_tensor(TaskType task_type, std::string name) {
  return commander_.have_midway_tensor(task_type, name);
}

// get a interim tensor by TaskType
std::shared_ptr<LockTensor> create_midway_tensor(TaskType task_type, 
                                          std::string name, 
                                          std::vector<int> dims, 
                                          ElementType element_type) {
  return commander_.create_midway_tensor(task_type, name, dims, element_type);
}

// copy dadt to tensor
void memcpy_to_tesnor(std::shared_ptr<LockTensor> tensor, const void *data, bool data_is_gpu) {
  commander_.memcpy_to_tesnor(tensor, data, data_is_gpu);
}

// copy dada from tesnor
void memcpy_from_tesnor(std::shared_ptr<LockTensor> tensor, void *data, bool data_is_gpu) {
  commander_.memcpy_from_tesnor(tensor, data, data_is_gpu);
}

}
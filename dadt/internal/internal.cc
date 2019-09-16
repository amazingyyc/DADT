#include "definition.h"
#include "internal.h"
#include "commander.h"

namespace dadt {

// the commander
dadt::Commander commander_;

// initialize dadt
void init(Config config) {
  commander_.init(config);
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

// timeline evet
void begin_timeline_event(const std::string &name, const std::string &event) {
  commander_.begin_timeline_event(name, event);
}

void end_timeline_event(const std::string &name, const std::string &event) {
  commander_.end_timeline_event(name, event);
}

#ifdef HAVE_NCCL
cudaEvent_t obtain_cuda_event() {
  commander_.obtain_cuda_event();
}
#endif

std::shared_ptr<LockTensor> obtain_midway_tensor(TaskType task_type, std::string name) {
  return commander_.obtain_midway_tensor(task_type, name);
}

// get a interim tensor by TaskType
std::shared_ptr<LockTensor> create_midway_tensor(TaskType task_type, 
                                          std::string name, 
                                          std::vector<int> dims, 
                                          ElementType element_type) {
  return commander_.create_midway_tensor(task_type, name, dims, element_type);
}

}
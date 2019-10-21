#ifndef TASK_H
#define TASK_H

#include <iostream>
#include <functional>
#include <memory>

#include "types.h"

namespace dadt {

class LockTensor;

// define the task type for now only support allreduce, broadcast
// shutdown is a special task type use for shut down whole system
const TaskType kShutDownTaskType  = -1;
const TaskType kAllReduceTaskType = 0;
const TaskType kBroadCastTaskType = 1;

#define DADT_SHUTDOWN_TASK_NAME "shutdown"

// a task struct
struct Task {
  // task type for now only support all reduce, broadcast, shutdown
  TaskType task_type;

  // every tensor/op have a unique name
  std::string name;

  // a tesnor used for this task, for shutdown it is null
  std::shared_ptr<LockTensor> tensor;

  // when finish call this function
  std::function<void()> done;
};

// a cell include task type, name, and corresponding tesnor size
struct TaskCell {
  TaskType task_type;

  std::string name;

  int num_bytes;
};

}

#endif
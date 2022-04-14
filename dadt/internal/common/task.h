#pragma once

#include <functional>
#include <memory>

#include "t/lock_tensor.h"

namespace dadt {

using TaskType = uint32_t;

struct TaskKey {
  TaskType type;
  uint32_t id;
};

// define TaskKey hash function
struct TaskKeyHash {
  std::size_t operator()(const TaskKey& k) const {
    return k.type ^ k.id;
  }
};

// define TaskKey equal function
struct TaskKeyEqual {
  bool operator()(const TaskKey& lhs, const TaskKey& rhs) const {
    return lhs.type == rhs.type && lhs.id == rhs.id;
  }
};

// define the task type for now only support allreduce, broadcast
// shutdown is a special task type use for shut down whole system
const TaskType kShutDownTaskType = 0;
const TaskType kBroadCastTaskType = 1;
const TaskType kAllReduceTaskType = 2;
const TaskType kCooAllReduceTaskType = 3;

const uint32_t kShutdowId = (uint32_t)(-1);

// a task struct
struct Task {
  // task type for now only support all reduce, broadcast, shutdown
  TaskType type;

  // Every tensor/op has a unique id and must same in every rank.
  // shutdown task always be: -1.
  uint32_t id;

  // a tensor used for this task, for shutdown it is null
  std::shared_ptr<LockTensor> l_tensor;

  // call this before executor
  std::function<void()> before;

  // when finish call this function
  std::function<void()> done;
};

}  // namespace dadt

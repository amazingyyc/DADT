#ifndef TASK_H
#define TASK_H

#include <iostream>
#include <functional>

namespace dadt {

class LockTensor;

typedef int TaskType;

// define the task type for now only support allreduce, broadcast
const TaskType DADTAllReduce = 0;
const TaskType DADTBroadCast = 1;

// a task struct
struct Task {
  // task type for now only support all reduce, broadcast
  TaskType task_type;

  // every tensor/op have a unique name
  std::string name;

  // a tesnor used for this task
  std::shared_ptr<LockTensor> tensor;

  // when finish call this function
  std::function<void()> done;
};

// define a hash map key
typedef std::tuple<TaskType, std::string> TaskKey;

struct TaskKeyHash {
  std::size_t operator()(const TaskKey& k) const {
    auto task_type = std::get<0>(k);
    auto name      = std::get<1>(k);

    return task_type ^ std::hash<std::string>{}(name);
  }
};

struct TaskKeyEqual {
  bool operator () (const TaskKey &lhs, const TaskKey &rhs) const {
    return std::get<0>(lhs) == std::get<0>(rhs) && std::get<1>(lhs) == std::get<1>(rhs);
  }
};

}

#endif
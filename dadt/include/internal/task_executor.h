#ifndef EXECUTOR_H
#define EXECUTOR_H

#include <iostream>
#include <vector>

#include "context.h"
#include "lock_tensor.h"


namespace dadt {

class ITaskExecutor {

public:
  // a executor may need a interim tensor to store the data and every executor may need different device tensor
  // li MPI broadcast need cpu tesnor
  virtual std::shared_ptr<LockTensor> get_interim_tensor(std::string name, std::vector<int> dims, ElementType element_type) = 0;

  // tasks will contain the task that have the some tasktype
  virtual void operator()(const Context &context, const std::vector<Task> &tasks) = 0;

};

}

#endif
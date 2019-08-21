#ifndef MPI_BROAD_CAST_EXECUTOR_H
#define MPI_BROAD_CAST_EXECUTOR_H

#include "device.h"
#include "task_executor.h"
#include "memory_buffer.h"

namespace dadt {

class MPIBroadCastExecutor: public ITaskExecutor {
private:
  // a cpu memory buffer
  MemoryBuffer buffer_;

public:
  MPIBroadCastExecutor();

  // if has already create a midway tensor
  std::shared_ptr<LockTensor> has_midway_tensor(std::string name) override;

  // a executor may need a interim tensor to store the data and every executor may need different device tensor
  // like MPI broadcast need cpu tesnor
  // the function is not thread safe
  std::shared_ptr<LockTensor> midway_tensor(std::string name, std::vector<int> dims, ElementType element_type) override;

  void operator()(const Context &context, const std::vector<Task> &tasks) override;
};

}

#endif

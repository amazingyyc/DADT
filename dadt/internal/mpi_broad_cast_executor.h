#ifndef MPI_BROAD_CAST_EXECUTOR_H
#define MPI_BROAD_CAST_EXECUTOR_H

#include "device.h"
#include "task_executor.h"
#include "memory_buffer.h"

namespace dadt {

class MPIBroadCastExecutor: public ITaskExecutor {
private:
  // cpu Device
  // MPIBroadCastExecutor will use CPU midway tensor
  std::shared_ptr<Device> cpu_device_;

public:
  MPIBroadCastExecutor(std::shared_ptr<Device> cpu_device);

  // if has already create a midway tensor
  std::shared_ptr<LockTensor> obtain_midway_tensor(std::string name) override;

  // a executor may need a interim tensor to store the data and every executor may need different device tensor
  // like MPI broadcast need cpu tesnor
  // the function is not thread safe
  std::shared_ptr<LockTensor> create_midway_tensor(std::string name, std::vector<int> dims, ElementType element_type) override;

  void operator()(const Context &context, const std::vector<Task> &tasks, std::shared_ptr<TimeLine> timeline) override;
};

}

#endif

#ifndef MPI_CUDA_BROAD_CAST_EXECUTOR_H
#define MPI_CUDA_BROAD_CAST_EXECUTOR_H

#include <iostream>
#include <unordered_map>
#include  <mutex>

#include "device.h"
#include "task_executor.h"
#include "memory_buffer.h"

namespace dadt {

// MPICUDABroadCastExecutor will use MPI to do broad cast, and the mpi must be GPU-aware.
class MPICUDABroadCastExecutor: public ITaskExecutor {
private:
  // NCCLAllReduceExecutor will GPU midway tensor
  std::shared_ptr<Device> gpu_device_;

public:
  MPICUDABroadCastExecutor(std::shared_ptr<Device> gpu_device);

  ~MPICUDABroadCastExecutor();

  std::shared_ptr<LockTensor> obtain_midway_tensor(std::string name) override;
  
  std::shared_ptr<LockTensor> create_midway_tensor(std::string name, std::vector<int> dims, ElementType element_type) override;

  void operator()(const Context &context, const std::vector<Task> &tasks, std::shared_ptr<TimeLine> timeline) override;
}

#endif
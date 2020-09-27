#ifndef MPI_CUDA_ALL_REDUCE_EXECUTOR_H
#define MPI_CUDA_ALL_REDUCE_EXECUTOR_H

#include <iostream>
#include <unordered_map>
#include  <mutex>

#include "device.h"
#include "task_executor.h"
#include "memory_buffer.h"

namespace dadt {

// different with MPIAllReduceExecutor
// MPICUDAAllReduceExecutor use GPU memory as midway tesnor
// and must traing on GPU
// and the mpi installed must GPU-aware
// ref:https://devblogs.nvidia.com/introduction-cuda-aware-mpi/
class MPICUDAAllReduceExecutor: public ITaskExecutor {
private:
  // MPICUDAAllReduceExecutor will GPU midway tensor
  std::shared_ptr<Device> gpu_device_;

  // tensor_pool_ will used in multi-thread
  std::mutex pool_mutex_;

  // allreduce will reuse the tensor, so use a map to store it
  // the tensor is GPU tensor
  std::unordered_map<std::string, std::shared_ptr<LockTensor>> tensor_pool_;

  // fusion buffer
  MemoryBuffer buffer_;

  // use a event wait cuda job finish
  cudaEvent_t finish_event_;

public:
  MPICUDAAllReduceExecutor(std::shared_ptr<Device> gpu_device, size_t buffer_size = 67108864);

  ~MPICUDAAllReduceExecutor();

  // whether already create a midway tensor
  std::shared_ptr<LockTensor> obtain_midway_tensor(std::string name) override;

  std::shared_ptr<LockTensor> create_midway_tensor(std::string name, std::vector<int> dims, ElementType element_type) override;

  void operator()(const Context &context, const std::vector<Task> &tasks, std::shared_ptr<TimeLine> timeline) override;
};

}

#endif

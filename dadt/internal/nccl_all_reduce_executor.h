#ifndef NCCL_ALL_REDUCE_EXECUTOR_H
#define NCCL_ALL_REDUCE_EXECUTOR_H

#include <iostream>
#include <unordered_map>
#include <mutex>

#include <cuda_runtime.h>
#include <nccl.h>

#include "device.h"
#include "task_executor.h"
#include "memory_buffer.h"

namespace dadt {

class NCCLAllReduceExecutor: public ITaskExecutor {
private:
  // tensor_pool_ will used in multi-thread
  std::mutex pool_mutex_;

  // nccl all reduce will resuse the midway tesnor
  std::unordered_map<std::string, std::shared_ptr<LockTensor>> tensor_pool_;

  // a gpu buffer
  MemoryBuffer buffer_;

  // gpu device id
  int gpu_device_id_;

  // use a event wait all reduce finish
  cudaEvent_t finish_event_;

public:
  // gpu_device_id: gpu device
  // buffer_size: buffer size
  NCCLAllReduceExecutor(int gpu_device_id, size_t buffer_size = 67108864);

  ~NCCLAllReduceExecutor();

  // if has already create a midway tensor
  std::shared_ptr<LockTensor> obtain_midway_tensor(std::string name) override;
  
  std::shared_ptr<LockTensor> create_midway_tensor(std::string name, std::vector<int> dims, ElementType element_type) override;

  void operator()(const Context &context, const std::vector<Task> &tasks) override;
};

}

#endif
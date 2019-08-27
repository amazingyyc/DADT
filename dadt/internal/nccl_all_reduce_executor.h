#ifndef NCCL_ALL_REDUCE_EXECUTOR_H
#define NCCL_ALL_REDUCE_EXECUTOR_H

#include <iostream>
#include <unordered_map>

#include <cuda_runtime.h>
#include <nccl.h>

#include "device.h"
#include "task_executor.h"
#include "memory_buffer.h"

namespace dadt {

class NCCLAllReduceExecutor: public ITaskExecutor {
private:
  // nccl all reduce will resuse the midway tesnor
  std::unordered_map<std::string, std::shared_ptr<LockTensor>> tensor_pool_;

  // a gpu buffer
  MemoryBuffer buffer_;

  // gpu device id
  int gpu_device_id_;

  // use a event wait all reduce finish
  cudaEvent finish_event_;

public:
  // gpu_device_id: gpu device
  // buffer_size: memory buffer size
  NCCLAllReduceExecutor(int gpu_device_id);

  ~NCCLAllReduceExecutor();

  // if has already create a midway tensor
  std::shared_ptr<LockTensor> have_midway_tensor(std::string name) override;
  
  std::shared_ptr<LockTensor> create_midway_tensor(std::string name, std::vector<int> dims, ElementType element_type) override;

  void operator()(const Context &context, const std::vector<Task> &tasks) override;
};

}
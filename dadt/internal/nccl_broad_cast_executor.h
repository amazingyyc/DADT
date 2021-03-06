#ifndef NCCL_BROAD_CAST_EXECUTOR_H
#define NCCL_BROAD_CAST_EXECUTOR_H

#include <iostream>
#include <unordered_map>

#include <cuda_runtime.h>
#include <nccl.h>

#include "device.h"
#include "task_executor.h"
#include "memory_buffer.h"

namespace dadt {

class NCCLBroadCastExecutor: public ITaskExecutor {
private:
  // NCCLAllReduceExecutor will GPU midway tensor
  std::shared_ptr<Device> gpu_device_;

  // use a event wait cuda stream finish
  cudaEvent_t finish_event_;

public:
  // gpu_device_id: gpu device
  NCCLBroadCastExecutor(std::shared_ptr<Device> gpu_device);

  ~NCCLBroadCastExecutor();

  bool is_cuda_midway_tensor() override;

  std::shared_ptr<LockTensor> obtain_midway_tensor(std::string name) override;
  
  std::shared_ptr<LockTensor> create_midway_tensor(std::string name, Shape shape, ElementType element_type) override;

  void operator()(const Context &context, const std::vector<Task> &tasks, std::shared_ptr<TimeLine> timeline) override;
};

}

#endif
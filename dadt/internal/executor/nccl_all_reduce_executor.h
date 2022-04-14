#pragma once

#include <cuda_runtime.h>
#include <nccl.h>

#include "common/buffer.h"
#include "common/device.h"
#include "executor/task_executor.h"

namespace dadt {

// nccl all reduce
class NCCLAllReduceExecutor : public ITaskExecutor {
private:
  // a gpu buffer
  Buffer buffer_;

  // use a event wait all reduce finish
  cudaEvent_t finish_event_;

public:
  // gpu_device_id: gpu device
  // buffer_size: buffer size
  NCCLAllReduceExecutor(Device* gpu_device, size_t buffer_size = 67108864);

  ~NCCLAllReduceExecutor();

public:
  void Do(const Context& context, const std::vector<Task>& tasks) override;
};

}  // namespace dadt

#pragma once

#include <cuda_runtime.h>
#include <nccl.h>

#include "common/buffer.h"
#include "common/device.h"
#include "executor/task_executor.h"

namespace dadt {

class NCCLBroadCastExecutor : public ITaskExecutor {
private:
  // use a event wait cuda stream finish
  cudaEvent_t finish_event_;

public:
  NCCLBroadCastExecutor();

  ~NCCLBroadCastExecutor();

  void Do(const Context& context, const std::vector<Task>& tasks) override;
};

}  // namespace dadt

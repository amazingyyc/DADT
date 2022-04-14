#pragma once

#include <cuda_runtime.h>
#include <nccl.h>

#include "common/buffer.h"
#include "common/device.h"
#include "executor/task_executor.h"

namespace dadt {

// nccl all reduce
class NCCLCooAllReduceExecutor : public ITaskExecutor {
private:
  // use a event wait all reduce finish
  cudaEvent_t finish_event_;

public:
  NCCLCooAllReduceExecutor();

  ~NCCLCooAllReduceExecutor();

private:
  Tensor DoImpl(const Context& context, const Tensor& coo_t);

public:
  void Do(const Context& context, const std::vector<Task>& tasks) override;
};

}  // namespace dadt

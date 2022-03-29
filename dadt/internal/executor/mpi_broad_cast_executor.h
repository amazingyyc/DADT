#pragma once

#include "common/device.h"
#include "executor/task_executor.h"

namespace dadt {

class MPIBroadCastExecutor : public ITaskExecutor {
public:
  MPIBroadCastExecutor();

  void Do(const Context& context, const std::vector<Task>& tasks) override;
};

}  // namespace dadt

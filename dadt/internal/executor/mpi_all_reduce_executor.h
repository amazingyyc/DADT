#pragma once

#include <iostream>
#include <unordered_map>

#include "common/buffer.h"
#include "common/device.h"
#include "executor/task_executor.h"

namespace dadt {

class MPIAllReduceExecutor : public ITaskExecutor {
private:
  // memory buffer, fusion tensor
  Buffer buffer_;

public:
  MPIAllReduceExecutor(Device* cpu_device, size_t buffer_size);

  void Do(const Context& context, const std::vector<Task>& tasks) override;
};

}  // namespace dadt

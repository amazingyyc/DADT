#pragma once

#include <iostream>
#include <unordered_map>

#include "common/buffer.h"
#include "common/device.h"
#include "executor/task_executor.h"
#include "t/shape.h"
#include "t/tensor.h"

namespace dadt {

// coo tensor ref:https://pytorch.org/docs/stable/sparse.html
class MPICooAllReduceExecutor : public ITaskExecutor {
public:
  MPICooAllReduceExecutor();

private:
  int64_t FlattenIndex(const Shape& shape, int64_t* dims);

  void FlattenIndexBack(int64_t index, const Shape& shape, int64_t* dims);

  Tensor DoImpl(const Context& context, const Tensor& coo_t);

public:
  void Do(const Context& context, const std::vector<Task>& tasks) override;
};

}  // namespace dadt

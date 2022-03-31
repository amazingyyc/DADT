#pragma once

#include <cinttypes>
#include <vector>

#ifdef HAVE_NCCL
#include <cuda_runtime.h>
#include <nccl.h>
#endif

#include "common/context.h"
#include "common/task.h"
#include "t/element_type.h"
#include "t/lock_tensor.h"

namespace dadt {

struct MergeUnit {
  size_t begin;
  size_t end;
};

class ITaskExecutor {
public:
  virtual ~ITaskExecutor() = default;

  // get mpi data type by element type
  MPI_Datatype MpiDataType(const Context& context, ElementType element_type);

  std::vector<int64_t> AllGatherV(const Context& context,
                                  const std::vector<int64_t>& vec);

#ifdef HAVE_NCCL
  ncclDataType_t NcclDataType(ElementType element_type);

  // Gather tensor from all rank and concate them together.
  // Like: output = concate(input[0], input[1], ..., input[rank], dim=0)
  Tensor AllGatherAndCatTensor(const Context& context, const Tensor& input);

#endif

  std::vector<MergeUnit> SplitTasks(const std::vector<Task>& tasks,
                                    size_t buffer_size);

  // tasks will contain the task that have the some tasktype
  virtual void Do(const Context& context, const std::vector<Task>& tasks) = 0;
};

}  // namespace dadt

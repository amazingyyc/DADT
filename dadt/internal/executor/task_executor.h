#pragma once

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

#ifdef HAVE_NCCL
  ncclDataType_t NcclDataType(const Context& context, ElementType element_type);
#endif

  // tasks will contain the task that have the some tasktype
  virtual void Do(const Context& context, const std::vector<Task>& tasks) = 0;
};

}  // namespace dadt

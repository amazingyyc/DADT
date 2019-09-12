#ifndef EXECUTOR_H
#define EXECUTOR_H

#include <iostream>
#include <vector>

#ifdef HAVE_NCCL
#include <cuda_runtime.h>
#include <nccl.h>
#endif

#include "context.h"
#include "lock_tensor.h"
#include "element_type.h"

namespace dadt {

class ITaskExecutor {
public:

  virtual ~ITaskExecutor() = default;

  // get mpi data type by element type
  MPI_Datatype mpi_data_type(const Context &context, ElementType element_type);

#ifdef HAVE_NCCL
  ncclDataType_t nccl_data_type(const Context &context, ElementType element_type);
#endif

  // obtain the midway tesnor may return nullptr
  virtual std::shared_ptr<LockTensor> obtain_midway_tensor(std::string name) = 0;

  // a executor may need a interim tensor to store the data and every executor may need different device tensor
  // like MPI broadcast need cpu tesnor
  virtual std::shared_ptr<LockTensor> create_midway_tensor(std::string name, std::vector<int> dims, ElementType element_type) = 0;

  // tasks will contain the task that have the some tasktype
  virtual void operator()(const Context &context, const std::vector<Task> &tasks) = 0;

};

}

#endif
#ifndef EXECUTOR_H
#define EXECUTOR_H

#include <iostream>
#include <vector>

#include "context.h"
#include "lock_tensor.h"
#include "element_type.h"

namespace dadt {

class ITaskExecutor {
public:
  // get mpi data type by element type
  MPI_Datatype mpi_data_type(const Context &context, ElementType element_type) {
    switch (element_type.id()) {
      case DType::Bool:
        return MPI_C_BOOL;
      case DType::Uint8:
        return MPI_UINT8_T;
      case DType::Int8:
        return MPI_INT8_T;
      case DType::Uint16:
        return MPI_UINT16_T;
      case DType::Int16:
        return MPI_INT16_T;
      case DType::Uint32:
        return MPI_UINT32_T;
      case DType::Int32:
        return MPI_INT32_T;
      case DType::Uint64:
        return MPI_UINT64_T;
      case DType::Int64:
        return MPI_INT64_T;
      case DType::Float16:
        return context.MPI_FLOAT16_T;
      case DType::Float32:
        return MPI_FLOAT;
      case DType::Float64:
        return MPI_DOUBLE;
      default:
        RUNTIME_ERROR("the dtype" << element_type.name() << " is not support in MPI");
    }
  }

  // if has already create a midway tensor
  virtual std::shared_ptr<LockTensor> has_midway_tensor(std::string name) = 0;

  // a executor may need a interim tensor to store the data and every executor may need different device tensor
  // li MPI broadcast need cpu tesnor
  virtual std::shared_ptr<LockTensor> midway_tensor(std::string name, std::vector<int> dims, ElementType element_type) = 0;

  // tasks will contain the task that have the some tasktype
  virtual void operator()(const Context &context, const std::vector<Task> &tasks) = 0;

};

}

#endif
#include "task_executor.h"

namespace dadt {

// get mpi data type by element type
MPI_Datatype ITaskExecutor::mpi_data_type(const Context &context, ElementType element_type) {
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

}
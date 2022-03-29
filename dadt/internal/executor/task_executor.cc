#include "executor/task_executor.h"

#include "common/exception.h"

namespace dadt {

// get mpi data type by element type
MPI_Datatype ITaskExecutor::MpiDataType(const Context& context,
                                        ElementType element_type) {
  switch (element_type.dtype) {
    case DType::kBool:
      return MPI_C_BOOL;
    case DType::kUint8:
      return MPI_UINT8_T;
    case DType::kInt8:
      return MPI_INT8_T;
    case DType::kUint16:
      return MPI_UINT16_T;
    case DType::kInt16:
      return MPI_INT16_T;
    case DType::kUint32:
      return MPI_UINT32_T;
    case DType::kInt32:
      return MPI_INT32_T;
    case DType::kUint64:
      return MPI_UINT64_T;
    case DType::kInt64:
      return MPI_INT64_T;
    case DType::kFloat16:
      return context.MPI_FLOAT16_T;
    case DType::kFloat32:
      return MPI_FLOAT;
    case DType::kFloat64:
      return MPI_DOUBLE;
    default:
      RUNTIME_ERROR("The dtype" << element_type.Name()
                                << " not support in MPI");
  }
}

#ifdef HAVE_NCCL
ncclDataType_t ITaskExecutor::NcclDataType(const Context& context,
                                           ElementType element_type) {
  switch (element_type.dtype) {
    case DType::kUint8:
      return ncclUint8;
    case DType::kInt8:
      return ncclInt8;
    case DType::kUint32:
      return ncclUint32;
    case DType::kInt32:
      return ncclInt32;
    case DType::kUint64:
      return ncclUint64;
    case DType::kInt64:
      return ncclInt64;
    case DType::kFloat16:
      return ncclFloat16;
    case DType::kFloat32:
      return ncclFloat32;
    case DType::kFloat64:
      return ncclFloat64;
    default:
      RUNTIME_ERROR("the dtype" << element_type.Name()
                                << " not support in NCCL");
  }
}
#endif

}  // namespace dadt

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

#ifdef HAVE_NCCL
ncclDataType_t ITaskExecutor::nccl_data_type(const Context &context, ElementType element_type) {
  switch (element_type.id()) {
    case DType::Uint8:
      return ncclUint8;
    case DType::Int8:
      return ncclInt8;
    case DType::Uint32:
      return ncclUint32;
    case DType::Int32:
      return ncclInt32;
    case DType::Uint64:
      return ncclUint64;
    case DType::Int64:
      return ncclInt64;
    case DType::Float16:
      return ncclFloat16;
    case DType::Float32:
      return ncclFloat32;
    case DType::Float64:
      return ncclFloat64;
    default:
      RUNTIME_ERROR("the dtype" << element_type.name() << " is not support in NCCL");
  }
}
#endif

// split tasks to MergeUnit
std::vector<MergeUnit> ITaskExecutor::split_tasks(const std::vector<Task> &tasks, size_t buffer_size) {
  std::vector<MergeUnit> merge_units;

  size_t tasks_size = tasks.size();

  for (size_t i = 0; i < tasks_size; ) {
    if (tasks[i].tensor->num_bytes() >= buffer_size) {
      MergeUnit unit;
      unit.begin = i;
      unit.end   = i + 1;

      merge_units.emplace_back(unit);

      i++;

      continue;
    }

    size_t current_buffer_size = 0;
    size_t j = i;

    for (; j < tasks_size; ++j) {
      if (current_buffer_size + tasks[j].tensor->num_bytes() > buffer_size) {
        break;
      }

      current_buffer_size += tasks[j].tensor->num_bytes();
    }

    MergeUnit unit;
    unit.begin = i;
    unit.end   = j;

    merge_units.emplace_back(unit);

    i = j;
  }

  return std::move(merge_units);
}

}
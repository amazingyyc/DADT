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

std::vector<int64_t> ITaskExecutor::MpiAllGatherV(
    const Context& context, const std::vector<int64_t>& vec) {
  std::vector<int64_t> all_vec;

  std::vector<uint64_t> counts(context.world_size);
  counts[context.world_rank] = vec.size();

  MPI_CALL(MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, counts.data(), 1,
                         MPI_UINT64_T, context.world_comm));

  uint64_t total_count = 0;
  for (auto i : counts) {
    total_count += i;
  }

  all_vec.resize(total_count);

  std::vector<int> recvcounts(context.world_size);
  std::vector<int> displs(context.world_size);

  for (int i = 0; i < context.world_size; ++i) {
    recvcounts[i] = counts[i];

    if (0 == i) {
      displs[i] = 0;
    } else {
      displs[i] = displs[i - 1] + recvcounts[i - 1];
    }
  }

  MPI_CALL(MPI_Allgatherv(vec.data(), (int)vec.size(), MPI_INT64_T,
                          all_vec.data(), recvcounts.data(), displs.data(),
                          MPI_INT64_T, context.world_comm));

  return all_vec;
}

#ifdef HAVE_NCCL
ncclDataType_t ITaskExecutor::NcclDataType(ElementType element_type) {
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

Tensor ITaskExecutor::NcclAllGatherAndCatTensor(const Context& context,
                                                const Tensor& input) {
  ARGUMENT_CHECK(input.shape().NDims() >= 1,
                 "NcclAllGatherAndCatTensor need tensor ndims >= 1");

  Shape input_shape = input.shape();

  // The tensor from all rank must has same shape except the first dimension.
  std::vector<int64_t> first_dims(context.world_size);
  first_dims[context.world_rank] = input_shape[0];

  MPI_CALL(MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, first_dims.data(),
                         1, MPI_INT64_T, context.world_comm));

  std::vector<int64_t> output_dims = input_shape.dims();
  output_dims[0] = 0;
  for (auto i : first_dims) {
    output_dims[0] += i;
  }

  auto element_type = input.element_type();

  Shape output_shape(output_dims);

  // We create a output tensor to accept the gather data.
  Tensor output = input.DynamicZero(output_shape, element_type);

  int64_t stride = input_shape.Stride(0);

  std::vector<int64_t> counts(context.world_size);
  for (int i = 0; i < context.world_size; ++i) {
    counts[i] = first_dims[i] * stride;
  }

  std::vector<int64_t> offsets(context.world_size);
  offsets[0] = 0;
  for (int i = 1; i < context.world_size; ++i) {
    offsets[i] =
        offsets[i - 1] + first_dims[i - 1] * stride * element_type.ByteWidth();
  }

  uint8_t* out_ptr = output.Data<uint8_t>();
  auto nccl_dtype = NcclDataType(element_type);

  NCCL_CALL(ncclGroupStart());
  for (int i = 0; i < context.world_size; ++i) {
    // Send
    NCCL_CALL(ncclSend(input.Ptr(), input.Size(), nccl_dtype, i,
                       context.nccl_comm, context.cuda_stream));

    // Recv
    NCCL_CALL(ncclRecv(out_ptr + offsets[i], counts[i], nccl_dtype, i,
                       context.nccl_comm, context.cuda_stream));
  }
  NCCL_CALL(ncclGroupEnd());

  return output;
}
#endif

std::vector<MergeUnit> ITaskExecutor::SplitTasks(const std::vector<Task>& tasks,
                                                 size_t buffer_size) {
  std::vector<MergeUnit> merge_units;

  for (size_t i = 0; i < tasks.size();) {
    if (tasks[i].l_tensor->tensor().NumBytes() >= buffer_size) {
      MergeUnit unit;
      unit.begin = i;
      unit.end = i + 1;

      merge_units.emplace_back(unit);
      i += 1;
    } else {
      size_t cur_size = 0;
      size_t j = i;

      for (; j < tasks.size(); ++j) {
        if (cur_size + tasks[j].l_tensor->tensor().NumBytes() > buffer_size) {
          break;
        }

        cur_size += tasks[j].l_tensor->tensor().NumBytes();
      }

      MergeUnit unit;
      unit.begin = i;
      unit.end = j;

      merge_units.emplace_back(unit);
      i = j;
    }
  }

  return merge_units;
}

}  // namespace dadt

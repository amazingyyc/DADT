#include "executor/mpi_coo_all_reduce_executor.h"

#ifdef HAVE_OPENMP
#include <omp.h>
#endif

#include "common/exception.h"
#include "t/shape.h"

namespace dadt {

MPICooAllReduceExecutor::MPICooAllReduceExecutor() {
}

int64_t MPICooAllReduceExecutor::FlattenIndex(const Shape& shape,
                                              int64_t* dims) {
  int64_t index = 0;
  for (int64_t i = 0; i < shape.NDims(); ++i) {
    index += shape.Stride(i) * dims[i];
  }

  return index;
}

void MPICooAllReduceExecutor::FlattenIndexBack(int64_t index,
                                               const Shape& shape,
                                               int64_t* dims) {
  for (int64_t i = 0; i < shape.NDims(); ++i) {
    dims[i] = index / shape.Stride(i);
    index %= shape.Stride(i);
  }
}

Tensor MPICooAllReduceExecutor::DoImpl(const Context& context,
                                       const Tensor& coo_t) {
  int64_t sparse_dim = coo_t.sparse_dim();
  int64_t dense_dim = coo_t.dense_dim();

  // indices shape:(nnz, sparse_dim)
  // values shape: (nnz,) + coo_t.shape[M : M + K]
  const Tensor& indices = coo_t.indices();
  const Tensor& values = coo_t.values();

  ARGUMENT_CHECK(indices.element_type().Is<int64_t>(),
                 "Coo's indices must int64.");
  ARGUMENT_CHECK(
      indices.IsContiguous() && values.IsContiguous(),
      "MPICooAllReduceExecutor need indices and values is contiguous");

  Shape shape = coo_t.shape();
  int64_t nnz = coo_t.nnz();

  // We assume the coo_t shape is:[row, col]
  // row = coo_t.shape[0:M].size()
  // col = coo_t.shape[M : M + K].size()
  int64_t col = 1;
  {
    for (int64_t i = sparse_dim; i < shape.NDims(); ++i) {
      col *= shape[i];
    }
  }

  // Sparse dims.
  Shape sparse_shape;
  {
    std::vector<int64_t> dims(sparse_dim);
    for (int64_t i = 0; i < sparse_dim; ++i) {
      dims[i] = shape[i];
    }

    sparse_shape = Shape(std::move(dims));
  }

  std::vector<int64_t> ids;
  ids.resize(nnz);
  {
    int64_t* ptr = indices.Data<int64_t>();
    for (int64_t i = 0; i < nnz; ++i) {
      ids[i] = FlattenIndex(sparse_shape, ptr + i * sparse_dim);
    }
  }

  std::vector<int64_t> all_ids = MpiAllGatherV(context, ids);

  // For now we has get all ids from all rank
  // Let's remove duplicate and sort.
  std::sort(all_ids.begin(), all_ids.end());
  all_ids.erase(std::unique(all_ids.begin(), all_ids.end()), all_ids.end());

  std::unordered_map<int64_t, size_t> target;
  for (size_t i = 0; i < all_ids.size(); ++i) {
    target[all_ids[i]] = i;
  }

  int64_t new_nnz = all_ids.size();

  // Create new indices.
  Tensor new_indices = indices.DynamicZero(Shape({new_nnz, sparse_dim}),
                                           ElementType::From<int64_t>());

  // Copy data from values to new values.
  std::vector<int64_t> new_values_dims = values.shape().dims();
  new_values_dims[0] = new_nnz;

  Tensor new_values =
      values.DynamicZero(Shape(new_values_dims), values.element_type());
  {
    uint8_t* ptr = (uint8_t*)values.Ptr();
    uint8_t* new_ptr = (uint8_t*)new_values.Ptr();

    size_t col_bytes = values.element_type().ByteWidth() * col;

    // The coo_t is coalesced so the ids is unique copy is enough.
#pragma omp parallel for
    for (int64_t i = 0; i < nnz; ++i) {
      memcpy(new_ptr + target[ids[i]] * col_bytes, ptr + i * col_bytes,
             col_bytes);
    }
  }

  auto mpi_dtype = MpiDataType(context, new_values.element_type());

  // Allreduce.
  MPI_CALL(MPI_Allreduce(MPI_IN_PLACE, new_values.Ptr(), (int)(new_nnz * col),
                         mpi_dtype, MPI_SUM, context.world_comm));

  // Reset new indices.
  {
    int64_t* ptr = new_indices.Data<int64_t>();
    for (size_t i = 0; i < all_ids.size(); ++i) {
      FlattenIndexBack(all_ids[i], sparse_shape, ptr + i * sparse_dim);
    }
  }

  return Tensor::CooTensor(new_indices, new_values, shape, true);
}

void MPICooAllReduceExecutor::Do(const Context& context,
                                 const std::vector<Task>& tasks) {
  if (tasks.empty()) {
    return;
  }

  for (const auto& task : tasks) {
    if (task.before) {
      task.before();
    }

    const Tensor& coo_t = task.l_tensor->tensor();

    ARGUMENT_CHECK(
        coo_t.IsCoo() && coo_t.IsCpu() && coo_t.is_coalesced(),
        "MPICooAllReduce need tensor is Coo, CPU and must be coalesced");

    Tensor new_coo_t = DoImpl(context, coo_t);
    task.l_tensor->ResetTensor(new_coo_t);

    if (task.done) {
      task.done();
    }
  }
}

}  // namespace dadt

#include "executor/nccl_coo_all_reduce_executor.h"

#include "common/exception.h"

namespace dadt {

NCCLCooAllReduceExecutor::NCCLCooAllReduceExecutor() {
  CUDA_CALL(cudaEventCreate(&finish_event_));
}

NCCLCooAllReduceExecutor::~NCCLCooAllReduceExecutor() {
  cudaEventDestroy(finish_event_);
}

Tensor NCCLCooAllReduceExecutor::DoImpl(const Context& context,
                                        const Tensor& coo_t) {
  Shape shape = coo_t.shape();

  // indices shape:(M, nnz)
  // values shape: (nnz,) + coo_t.shape[M : M + K]

  // Transpose to: (nnz, M)
  Tensor indices_t = coo_t.Indices().Transpose(0, 1);

  // Concate with other rank shape: (new_nnz, M)
  Tensor new_indices = AllGatherAndCatTensor(context, indices_t);

  // Transpose back, shape: (M, new_nnz)
  new_indices = new_indices.Transpose(0, 1);

  // Concate values, shape: (new_nnz, coo_t.shape[M : M + K])
  Tensor new_values = AllGatherAndCatTensor(context, coo_t.Values());

  // Create a new Coo tensor.
  return coo_t.DynamicCoo(new_indices, new_values, shape).Coalesce();
}

void NCCLCooAllReduceExecutor::Do(const Context& context,
                                  const std::vector<Task>& tasks) {
  if (tasks.empty()) {
    return;
  }

  // All GPU tensor operator need bind to a CUDA stream, the TensorImpl (like
  // torch::tensor) maybe bind it's own CUDA stream. So before we do our
  // operator we need change the CUDA stream to context.cuda_stream.
  auto _ = tasks[0].l_tensor->tensor().DynamicCudaStreamGuard(
      context.cuda_stream, context.gpu_device_id);

  for (const auto& task : tasks) {
    if (task.before) {
      task.before();
    }

    const Tensor& coo_t = task.l_tensor->tensor();

    ARGUMENT_CHECK(
        coo_t.IsCoo() && coo_t.IsCuda() && coo_t.IsCoalesced(),
        "MPICooAllReduce need tensor is Coo, GPU and must be coalesced");

    Tensor new_coo_t = DoImpl(context, coo_t);
    task.l_tensor->ResetTensor(new_coo_t);

    // Wait all reduce finish
    CUDA_CALL(cudaEventRecord(finish_event_, context.cuda_stream));
    CUDA_CALL(cudaEventSynchronize(finish_event_));

    if (task.done) {
      task.done();
    }
  }
}

}  // namespace dadt

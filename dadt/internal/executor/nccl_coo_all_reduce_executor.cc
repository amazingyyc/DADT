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

  // indices shape:(nnz, sparse_dim)
  // values shape: (nnz,) + coo_t.shape[M : M + K]

  // Concate with other rank shape: (new_nnz, M)
  Tensor new_indices = NcclAllGatherAndCatTensor(context, coo_t.indices());

  // Concate values, shape: (new_nnz, coo_t.shape[M : M + K])
  Tensor new_values = NcclAllGatherAndCatTensor(context, coo_t.values());

  // Create a new Coo tensor.
  return Tensor::CooTensor(new_indices, new_values, shape, false);
}

void NCCLCooAllReduceExecutor::Do(const Context& context,
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
        coo_t.IsCoo() && coo_t.IsCuda() && coo_t.is_coalesced(),
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

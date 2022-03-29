#include "executor/nccl_broad_cast_executor.h"

#include "common/exception.h"

namespace dadt {

NCCLBroadCastExecutor::NCCLBroadCastExecutor() {
  // use a event wait broad cast finish
  CUDA_CALL(cudaEventCreate(&finish_event_));
}

NCCLBroadCastExecutor::~NCCLBroadCastExecutor() {
  // CUDA_CALL(cudaEventDestroy(finish_event_));
  cudaEventDestroy(finish_event_);
}

void NCCLBroadCastExecutor::Do(const Context& context,
                               const std::vector<Task>& tasks) {
  // call before callback
  for (auto& task : tasks) {
    if (task.before) {
      task.before();
    }
  }

  // for broad cast we will broad one by one
  for (auto& task : tasks) {
    ARGUMENT_CHECK(task.l_tensor->tensor().IsCuda(),
                   "NCCLBroadCastExecutor only support GPU tensor");

    void* sendbuf = task.l_tensor->tensor().Ptr();
    int count = task.l_tensor->tensor().Size();

    auto nccl_dtype =
        NcclDataType(context, task.l_tensor->tensor().element_type());

    NCCL_CALL(ncclBroadcast(sendbuf, sendbuf, count, nccl_dtype, 0,
                            context.nccl_comm, context.cuda_stream));
  }

  // wait cuda stream finish
  CUDA_CALL(cudaEventRecord(finish_event_, context.cuda_stream));
  CUDA_CALL(cudaEventSynchronize(finish_event_));

  // callback tensor
  for (auto& task : tasks) {
    if (task.done) {
      task.done();
    }
  }
}

}  // namespace dadt
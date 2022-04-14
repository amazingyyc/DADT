#include "nccl_all_reduce_executor.h"

#include "common/exception.h"

namespace dadt {

NCCLAllReduceExecutor::NCCLAllReduceExecutor(Device* gpu_device,
                                             size_t buffer_size)
    : buffer_(gpu_device) {
  buffer_.Reserve(buffer_size);

  // use a event wait all reduce finish
  CUDA_CALL(cudaEventCreate(&finish_event_));
}

NCCLAllReduceExecutor::~NCCLAllReduceExecutor() {
  // CUDA_CALL(cudaEventDestroy(finish_event_));
  cudaEventDestroy(finish_event_);
}

void NCCLAllReduceExecutor::Do(const Context& context,
                               const std::vector<Task>& tasks) {
  if (tasks.empty()) {
    return;
  }

  auto element_type = tasks[0].l_tensor->tensor().element_type();

  // check the element type and tensor device type
  for (auto& task : tasks) {
    ARGUMENT_CHECK(task.l_tensor->tensor().IsCuda(),
                   "NCCL all reduce need tensor is GPU");
    ARGUMENT_CHECK(element_type == task.l_tensor->tensor().element_type(),
                   "NCCL all reduce need all tensor has same element type");
  }

  auto nccl_dtype = NcclDataType(element_type);

  // iterator all task collect it by the buffer size
  auto merge_units = SplitTasks(tasks, buffer_.size());

  for (auto& unit : merge_units) {
    for (size_t i = unit.begin; i < unit.end; ++i) {
      if (tasks[i].before) {
        tasks[i].before();
      }
    }

    void* recvbuf = nullptr;
    size_t count = 0;

    if (unit.begin + 1 == unit.end) {
      recvbuf = tasks[unit.begin].l_tensor->tensor().Ptr();
      count = tasks[unit.begin].l_tensor->tensor().Size();
    } else {
      // copy tensor to buffer
      size_t offset = 0;

      for (size_t i = unit.begin; i < unit.end; ++i) {
        // copy memory async
        CUDA_CALL(cudaMemcpyAsync(
            buffer_.ptr(offset), tasks[i].l_tensor->tensor().Ptr(),
            tasks[i].l_tensor->tensor().NumBytes(), cudaMemcpyDeviceToDevice,
            context.cuda_stream));

        offset += tasks[i].l_tensor->tensor().NumBytes();
        count += tasks[i].l_tensor->tensor().Size();
      }

      recvbuf = buffer_.ptr();
    }

    NCCL_CALL(ncclAllReduce(recvbuf, recvbuf, count, nccl_dtype, ncclSum,
                            context.nccl_comm, context.cuda_stream));

    // copy back
    if (unit.begin + 1 != unit.end) {
      size_t offset = 0;

      for (size_t i = unit.begin; i < unit.end; ++i) {
        CUDA_CALL(cudaMemcpyAsync(
            tasks[i].l_tensor->tensor().Ptr(), buffer_.ptr(offset),
            tasks[i].l_tensor->tensor().NumBytes(), cudaMemcpyDeviceToDevice,
            context.cuda_stream));

        offset += tasks[i].l_tensor->tensor().NumBytes();
      }
    }

    // wait cuda stream finish
    CUDA_CALL(cudaEventRecord(finish_event_, context.cuda_stream));
    CUDA_CALL(cudaEventSynchronize(finish_event_));

    for (size_t i = unit.begin; i < unit.end; ++i) {
      if (tasks[i].done) {
        tasks[i].done();
      }
    }
  }
}

}  // namespace dadt

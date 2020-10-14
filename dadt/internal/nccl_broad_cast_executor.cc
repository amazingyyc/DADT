#include "definition.h"
#include "nccl_broad_cast_executor.h"

namespace dadt {

NCCLBroadCastExecutor::NCCLBroadCastExecutor(std::shared_ptr<Device> gpu_device): gpu_device_(gpu_device) {
  // use a event wait broad cast finish
  CUDA_CALL(cudaEventCreate(&finish_event_));
}

NCCLBroadCastExecutor::~NCCLBroadCastExecutor() {
  CUDA_CALL(cudaEventDestroy(finish_event_));
}

bool NCCLBroadCastExecutor::is_cuda_midway_tensor() {
  return true;
}

std::shared_ptr<LockTensor> NCCLBroadCastExecutor::obtain_midway_tensor(std::string name) {
  return nullptr;
}

std::shared_ptr<LockTensor> NCCLBroadCastExecutor::create_midway_tensor(std::string name, Shape shape, ElementType element_type) {
  auto storage = TensorStorage::create(gpu_device_, shape.size() * element_type.byte_width());

  auto tensor = std::make_shared<LockTensor>(storage, 0, shape, element_type, name, LockTensorStatus::kInFetch);

  return tensor;
}

void NCCLBroadCastExecutor::operator()(const Context &context, const std::vector<Task> &tasks, std::shared_ptr<TimeLine> timeline) {
  // begin broad cast timeline
  // if (context.enable_timeline.load()) {
  //   timeline->begin(tasks, kDoBroadCastEvent);
  // }

  // call before callback
  for (auto &task : tasks) {
    if (task.before) {
      task.before();
    }
  }

  // for broad cast we will broad one by one
  for (auto &task : tasks) {
    ARGUMENT_CHECK(task.tensor->is_cuda(), "NCCLBroadCastExecutor only support GPU tensor");

    void *sendbuf = task.tensor->ptr();
    int count = task.tensor->size();

    auto nccl_dtype = nccl_data_type(context, task.tensor->element_type());

    NCCL_CALL(ncclBroadcast(sendbuf, sendbuf, count, nccl_dtype, 0, context.nccl_comm, context.cuda_stream));
  }

  // wait cuda stream finish
  CUDA_CALL(cudaEventRecord(finish_event_, context.cuda_stream));
  CUDA_CALL(cudaEventSynchronize(finish_event_));

  // callback tensor
  for (auto &task : tasks) {
    if (task.done) {
      task.done();
    }
  }

  // end broad cast timeline
  // if (context.enable_timeline.load()) {
  //   timeline->end(tasks, kDoBroadCastEvent);
  // }
}

}
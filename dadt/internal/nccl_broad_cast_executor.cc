#include "definition.h"
#include "nccl_broad_cast_executor.h"

namespace dadt {

NCCLBroadCastExecutor::NCCLBroadCastExecutor(int gpu_device_id): gpu_device_id_(gpu_device_id) {
  // use a event wait broad cast finish
  CUDA_CALL(cudaEventCreate(&finish_event_));
}

NCCLBroadCastExecutor::~NCCLBroadCastExecutor() {
  CUDA_CALL(cudaEventDestroy(finish_event_));
}

std::shared_ptr<LockTensor> NCCLBroadCastExecutor::have_midway_tensor(std::string name) {
  return std::shared_ptr<LockTensor>();
}

std::shared_ptr<LockTensor> NCCLBroadCastExecutor::create_midway_tensor(std::string name, std::vector<int> dims, ElementType element_type) {
  // create GPU tensor
  auto device = get_gpu_device(gpu_device_id_);

  Shape shape(dims);

  auto storage = TensorStorage::create(device, shape.size() * element_type.byte_width());

  auto tensor = std::make_shared<LockTensor>(storage, 0, shape, element_type, name, LockTensorStatus::InFill);

  return tensor;
}

void NCCLBroadCastExecutor::operator()(const Context &context, const std::vector<Task> &tasks) {
  // for broad cast we will broad one by one
  for (auto &task : tasks) {
    ARGUMENT_CHECK(DeviceType::GPU == task.tensor->device()->device_type(), "NCCLBroadCastExecutor only support GPU tensor");

    void *sendbuf = task.tensor->ptr();
    int count = task.tensor->size();

    auto nccl_dtype = nccl_data_type(context, task.tensor->element_type());

    NCCL_CALL(ncclBroadcast(sendbuf, sendbuf, count, nccl_dtype, 0, context.nccl_comm, context.cuda_stream));
  }

  // wait cuda stream finish
  CUDA_CALL(cudaEventRecord(finish_event_, context.cuda_stream));
  CUDA_CALL(cudaEventSynchronize(finish_event_));
}

}
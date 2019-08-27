#include "nccl_all_reduce_executor.h"

namespace dadt {

NCCLAllReduceExecutor::NCCLAllReduceExecutor(int gpu_device_id)
: gpu_device_id_(gpu_device_id), buffer_(get_gpu_device(gpu_device_id)) {
  // when init reserve a 64mb buffer
  size_t buffer_size = 64 * 1024 * 1024;
  buffer_.reserve(buffer_size);

  // use a event wait all reduce finish
  CUDA_CALL(cudaEventCreate(&finish_event_));
}

NCCLAllReduceExecutor::~NCCLAllReduceExecutor() {
  CUDA_CALL(cudaEventDestroy(finish_event_));
}

// if has already create a midway tensor
std::shared_ptr<LockTensor> NCCLAllReduceExecutor::have_midway_tensor(std::string name) {
  if (tensor_pool_.find(name) != tensor_pool_.end()) {
    return tensor_pool_[name];
  }

  return std::shared_ptr<LockTensor>();
}

std::shared_ptr<LockTensor> NCCLAllReduceExecutor::create_midway_tensor(std::string name, std::vector<int> dims, ElementType element_type) {
  if (tensor_pool_.find(name) != tensor_pool_.end()) {
    // have created the tensor, resue it
    auto tensor = tensor_pool_[name];

    ARGUMENT_CHECK(tensor->shape() == Shape(dims) && tensor->element_type() == element_type, "get tesnor error!");

    return tensor;
  }

  // create GPU tensor
  auto device = get_gpu_device(gpu_device_id_);

  Shape shape(dims);

  auto storage = TensorStorage::create(device, shape.size() * element_type.byte_width());

  // the tensor in allreduce inited status is waitforfetch
  auto tensor = std::make_shared<LockTensor>(storage, 0, shape, element_type, name, LockTensorStatus::WaitForFetch);
  
  tensor_pool_[name] = tensor;

  return tensor;
}

void NCCLAllReduceExecutor::operator()(const Context &context, const std::vector<Task> &tasks) {
  auto element_type = tasks[0].tensor->element_type();

  ARGUMENT_CHECK(element_type.is<half>() || element_type.is<float>() || element_type.is<double>(), "NCCL all reduce only support half/float/double");

  // check the element type and tensor device type
  for (auto &task : tasks) {
    ARGUMENT_CHECK(DeviceType::GPU == task.tensor->device()->device_type(), "NCCL all reduce need tensor is GPU");
    ARGUMENT_CHECK(element_type == task.tensor->element_type(), "NCCL all reduce only support half/float/double");
  }

  void *recvbuff = nullptr;
  size_t count = 0;

  if (tasks.size() > 1) {
    size_t memory_size = 0;

    for (auto &task : tasks) {
      count += task.tensor->size();
      memory_size += task.tensor->num_bytes();
    }

    // reserve enough memory
    buffer_.reserve(memory_size);

    // copy tensor to buffer
    size_t offset = 0;

    for (auto &task : tasks) {
      // copy memory async
      CUDA_CALL(cudaMemcpyAsync(buffer_.ptr(offset), task.tensor->ptr(), task.tensor->num_bytes(), cudaMemcpyDeviceToDevice, context.cuda_stream));

      offset += task.tensor->num_bytes();
    }

    recvbuf = buffer_.ptr();
  } else {
    recvbuf = tasks[0].tensor->ptr();
    count   = tasks[0].tensor->size();
  }

  // all reduce
  auto nccl_dtype = nccl_data_type(context, tasks[0].tensor->element_type());

  NCCL_CALL(ncclAllReduce(recvbuff, recvbuff, count, nccl_dtype, ncclSum, context.nccl_comm, context.cuda_stream));

  // copy back
  if (tasks.size() > 1) {
    size_t offset = 0;

    for (auto &task : tasks) {
      CUDA_CALL(cudaMemcpyAsync(task.tensor->ptr(), buffer_.ptr(offset), task.tensor->num_bytes(), cudaMemcpyDeviceToDevice, context.cuda_stream));
      
      offset += task.tensor->num_bytes();
    }
  }

  // wait cuda stream finish
  CUDA_CALL(cudaEventRecord(finish_event_, context.cuda_stream));
  CUDA_CALL(cudaEventSynchronize(finish_event_));
}

}
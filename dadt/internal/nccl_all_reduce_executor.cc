#include "definition.h"
#include "nccl_all_reduce_executor.h"

namespace dadt {

NCCLAllReduceExecutor::NCCLAllReduceExecutor(int gpu_device_id, size_t buffer_size)
: gpu_device_id_(gpu_device_id), buffer_(get_gpu_device(gpu_device_id)) {
  // when init reserve a 64MB buffer
  if (buffer_size <= 0) {
    buffer_size = 64 * 1024 * 1024;
  }

  buffer_.reserve(buffer_size);

  // use a event wait all reduce finish
  CUDA_CALL(cudaEventCreate(&finish_event_));
}

NCCLAllReduceExecutor::~NCCLAllReduceExecutor() {
  CUDA_CALL(cudaEventDestroy(finish_event_));
}

std::shared_ptr<LockTensor> NCCLAllReduceExecutor::obtain_midway_tensor(std::string name) {
  // add lock
  std::unique_lock<std::mutex> lock(pool_mutex_);

  if (tensor_pool_.find(name) != tensor_pool_.end()) {
    return tensor_pool_[name];
  }

  lock.unlock();

  return std::shared_ptr<LockTensor>();
}

std::shared_ptr<LockTensor> NCCLAllReduceExecutor::create_midway_tensor(std::string name, std::vector<int> dims, ElementType element_type) {
  std::unique_lock<std::mutex> lock(pool_mutex_);

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

  // iterator all task collect it by the buffer size
  auto merge_units = split_tasks(tasks, buffer_.size());

  for (auto &unit : merge_units) {
    void *recvbuf = nullptr;
    size_t count = 0;

    if (unit.begin + 1 == unit.end) {
      recvbuf = tasks[unit.begin].tensor->ptr();
      count = tasks[unit.begin].tensor->size();
    } else {
      // copy tensor to buffer
      size_t offset = 0;

      for (size_t i = unit.begin; i < unit.end; ++i) {
        // copy memory async
        CUDA_CALL(cudaMemcpyAsync(buffer_.ptr(offset), 
                                  tasks[i].tensor->ptr(), 
                                  tasks[i].tensor->num_bytes(), 
                                  cudaMemcpyDeviceToDevice, 
                                  context.cuda_stream));

        offset += tasks[i].tensor->num_bytes();

        count += tasks[i].tensor->size();
      }

      recvbuf = buffer_.ptr();
    }

    // all reduce
    auto nccl_dtype = nccl_data_type(context, tasks[unit.begin].tensor->element_type());

    NCCL_CALL(ncclAllReduce(recvbuf, recvbuf, count, nccl_dtype, ncclSum, context.nccl_comm, context.cuda_stream));

    // copy back
    if (unit.begin + 1 != unit.end) {
      size_t offset = 0;

      for (size_t i = unit.begin; i < unit.end; ++i) {
        CUDA_CALL(cudaMemcpyAsync(tasks[i].tensor->ptr(), 
                                  buffer_.ptr(offset), 
                                  tasks[i].tensor->num_bytes(), 
                                  cudaMemcpyDeviceToDevice,
                                  context.cuda_stream));
      
        offset += tasks[i].tensor->num_bytes();
      }
    }

    // wait cuda stream finish
    CUDA_CALL(cudaEventRecord(finish_event_, context.cuda_stream));
    CUDA_CALL(cudaEventSynchronize(finish_event_));
    
    // callback tensor
    for (size_t i = unit.begin; i < unit.end; ++i) {
      tasks[i].done();
    }
  }
}

}
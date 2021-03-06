#include "definition.h"
#include "nccl_all_reduce_executor.h"

namespace dadt {

NCCLAllReduceExecutor::NCCLAllReduceExecutor(std::shared_ptr<Device> gpu_device, size_t buffer_size)
: gpu_device_(gpu_device), buffer_(gpu_device) {

  buffer_.reserve(buffer_size);

  // use a event wait all reduce finish
  CUDA_CALL(cudaEventCreate(&finish_event_));
}

NCCLAllReduceExecutor::~NCCLAllReduceExecutor() {
  CUDA_CALL(cudaEventDestroy(finish_event_));
}

void NCCLAllReduceExecutor::insert_midway_tensor(std::string name, std::shared_ptr<LockTensor> tensor) {
  SpinLockHandler handler(pool_locker_);

  tensor_pool_[name] = tensor;
}

bool NCCLAllReduceExecutor::is_cuda_midway_tensor() {
  return true;
}

std::shared_ptr<LockTensor> NCCLAllReduceExecutor::obtain_midway_tensor(std::string name) {
  SpinLockHandler handler(pool_locker_);

  if (tensor_pool_.find(name) != tensor_pool_.end()) {
    return tensor_pool_[name];
  }

  return nullptr;
}

std::shared_ptr<LockTensor> NCCLAllReduceExecutor::create_midway_tensor(std::string name, Shape shape, ElementType element_type) {
  SpinLockHandler handler(pool_locker_);

  if (tensor_pool_.find(name) != tensor_pool_.end()) {
    // have created the tensor, resue it
    auto tensor = tensor_pool_[name];

    ARGUMENT_CHECK(tensor->shape() == shape && tensor->element_type() == element_type, "get tesnor error!");

    return tensor;
  }

  auto storage = TensorStorage::create(gpu_device_, shape.size() * element_type.byte_width());

  // the tensor in allreduce inited status is kWaitForFetch
  auto tensor = std::make_shared<LockTensor>(storage, 0, shape, element_type, name, LockTensorStatus::kWaitForFetch);

  tensor_pool_[name] = tensor;

  return tensor;
}

void NCCLAllReduceExecutor::operator()(const Context &context, const std::vector<Task> &tasks, std::shared_ptr<TimeLine> timeline) {
  auto element_type = tasks[0].tensor->element_type();

  ARGUMENT_CHECK(element_type.is<half>() || element_type.is<float>() || element_type.is<double>(), "NCCL all reduce only support half/float/double");

  // check the element type and tensor device type
  for (auto &task : tasks) {
    ARGUMENT_CHECK(task.tensor->is_cuda(), "NCCL all reduce need tensor is GPU");
    ARGUMENT_CHECK(element_type == task.tensor->element_type(), "NCCL all reduce only support half/float/double");
  }

  // begin allreduce timeline
  // if (context.enable_timeline.load()) {
  //   timeline->begin(tasks, kDoAllReduceEvent);
  // }

  // iterator all task collect it by the buffer size
  auto merge_units = split_tasks(tasks, buffer_.size());

  for (auto &unit : merge_units) {
    // before do allreduce, should call before function of task
    for (size_t i = unit.begin; i < unit.end; ++i) {
      if (tasks[i].before) {
        tasks[i].before();
      }
    }

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
      if (tasks[i].done) {
        tasks[i].done();
      }
    }

    // timeline
    // if (context.enable_timeline.load()) {
    //   for (size_t i = unit.begin; i < unit.end; ++i) {
    //     timeline->end(tasks[i].name, kDoAllReduceEvent);
    //   }
    // }
  }
}

}
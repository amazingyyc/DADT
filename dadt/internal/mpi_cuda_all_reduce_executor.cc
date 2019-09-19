#include "mpi_cuda_all_reduce_executor.h"

namespace dadt {

MPICUDAAllReduceExecutor::MPICUDAAllReduceExecutor(std::shared_ptr<Device> gpu_device, size_t buffer_size)
  :gpu_device_(gpu_device), buffer_(gpu_device) {
  
  // reserve enough buffer
  buffer_.reserve(buffer_size);

  // create a cuda event, to wait memory copy finish
  CUDA_CALL(cudaEventCreate(&finish_event_));
}

MPICUDAAllReduceExecutor::~MPICUDAAllReduceExecutor() {
  CUDA_CALL(cudaEventDestroy(finish_event_));
}

// whether already create a midway tensor
std::shared_ptr<LockTensor> MPICUDAAllReduceExecutor::obtain_midway_tensor(std::string name) {
  std::unique_lock<std::mutex> lock(pool_mutex_);

  if (tensor_pool_.find(name) != tensor_pool_.end()) {
    return tensor_pool_[name];
  }

  return std::shared_ptr<LockTensor>();
}

std::shared_ptr<LockTensor> MPICUDAAllReduceExecutor::create_midway_tensor(std::string name, std::vector<int> dims, ElementType element_type) {
  std::unique_lock<std::mutex> lock(pool_mutex_);

  if (tensor_pool_.find(name) != tensor_pool_.end()) {
    auto tensor = tensor_pool_[name];

    ARGUMENT_CHECK(tensor->shape() == Shape(dims) && tensor->element_type() == element_type, "get tesnor error!");

    return tensor;
  }

  Shape shape(dims);

  auto storage = TensorStorage::create(gpu_device_, shape.size() * element_type.byte_width());

  // the tensor in allreduce inited status is kWaitForFetch
  auto tensor = std::make_shared<LockTensor>(storage, 0, shape, element_type, name, LockTensorStatus::kWaitForFetch);

  tensor_pool_[name] = tensor;

  return tensor;
}

void MPICUDAAllReduceExecutor::operator()(const Context &context, const std::vector<Task> &tasks, std::shared_ptr<TimeLine> timeline) {
  auto element_type = tasks[0].tensor->element_type();

  ARGUMENT_CHECK(element_type.is<float>() || element_type.is<double>(), "mpi cuda all reduce executor only support float/double.");

  // check the element type and tensor device type
  for (auto &task : tasks) {
    ARGUMENT_CHECK(DeviceType::GPU == task.tensor->device()->device_type(), "mpi cuda all reduce executor need tensor is GPU");
    ARGUMENT_CHECK(element_type == task.tensor->element_type(), "mpi cuda all reduce executor only support float/double");
  }

  // timeline
  if (context.enable_timeline.load()) {
    timeline->begin(tasks, kDoAllReduceEvent);
  }

  // split task
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

      // wait memory copy finish or will not start mpi all reduce
      CUDA_CALL(cudaEventRecord(finish_event_, context.cuda_stream));
      CUDA_CALL(cudaEventSynchronize(finish_event_));
    }

    // use mpi to do all reduce
    auto mpi_dtype = mpi_data_type(context, tasks[0].tensor->element_type());

    MPI_CALL(MPI_Allreduce(MPI_IN_PLACE, recvbuf, count, mpi_dtype, MPI_SUM, context.world_comm));

    // copy back to midway tensor
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

      // copy back
      CUDA_CALL(cudaEventRecord(finish_event_, context.cuda_stream));
      CUDA_CALL(cudaEventSynchronize(finish_event_));
    }

    // midway tensor callback
    for (size_t i = unit.begin; i < unit.end; ++i) {
      tasks[i].done();
    }

    // timeline
    if (context.enable_timeline.load()) {
      for (size_t i = unit.begin; i < unit.end; ++i) {
        timeline->end(tasks[i].name, kDoAllReduceEvent);
      }
    }
  }
}

}
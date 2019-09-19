#include "device.h"
#include "mpi_all_reduce_executor.h"

namespace dadt {

MPIAllReduceExecutor::MPIAllReduceExecutor(std::shared_ptr<Device> cpu_device, size_t buffer_size)
  : cpu_device_(cpu_device), buffer_(cpu_device) {
  buffer_.reserve(buffer_size);
}

std::shared_ptr<LockTensor> MPIAllReduceExecutor::obtain_midway_tensor(std::string name) {
  std::unique_lock<std::mutex> lock(pool_mutex_);

  if (tensor_pool_.find(name) != tensor_pool_.end()) {
    return tensor_pool_[name];
  }

  return std::shared_ptr<LockTensor>();
}

std::shared_ptr<LockTensor> MPIAllReduceExecutor::create_midway_tensor(std::string name, std::vector<int> dims, ElementType element_type) {
  std::unique_lock<std::mutex> lock(pool_mutex_);

  if (tensor_pool_.find(name) != tensor_pool_.end()) {
    // have created the tensor, resue it
    auto tensor = tensor_pool_[name];

    ARGUMENT_CHECK(tensor->shape() == Shape(dims) && tensor->element_type() == element_type, "get tensor error!");

    return tensor;
  }

  Shape shape(dims);

  // create a CPU tensor
  auto storage = TensorStorage::create(cpu_device_, shape.size() * element_type.byte_width());

  // the tensor in allreduce inited status is kWaitForFetch
  auto tensor = std::make_shared<LockTensor>(storage, 0, shape, element_type, name, LockTensorStatus::kWaitForFetch);
  
  tensor_pool_[name] = tensor;

  return tensor;
}

void MPIAllReduceExecutor::operator()(const Context &context, const std::vector<Task> &tasks, std::shared_ptr<TimeLine> timeline) {
  // mpi all reduce only support cpu tensor and float/double
  auto element_type = tasks[0].tensor->element_type();

  ARGUMENT_CHECK(element_type.is<float>() || element_type.is<double>(), "MPIAllReduceExecutor only support float/double");

  for (auto &task : tasks) {
    ARGUMENT_CHECK(DeviceType::CPU == task.tensor->device()->device_type() && 
                  element_type == task.tensor->element_type(), 
                  "mpi all reduce only support cpu tensor, element type must be float/double")
  }

  // begin allreduce timeline
  if (context.enable_timeline.load()) {
    timeline->begin(tasks, kDoAllReduceEvent);
  }

  auto merge_units = split_tasks(tasks, buffer_.size());

  for (auto &unit : merge_units) {
    void *recvbuf = nullptr;
    int count = 0;

    if (unit.begin + 1 == unit.end) {
      recvbuf = tasks[unit.begin].tensor->ptr();
      count = tasks[unit.begin].tensor->size();
    } else {
      // copy tensor to buffer
      size_t offset = 0;

      for (size_t i = unit.begin; i < unit.end; ++i) {
        std::memcpy(buffer_.ptr(offset), tasks[i].tensor->ptr(), tasks[i].tensor->num_bytes());

        offset += tasks[i].tensor->num_bytes();
        count += tasks[i].tensor->size();
      }

      recvbuf = buffer_.ptr();
    }

    auto mpi_dtype = mpi_data_type(context, tasks[0].tensor->element_type());

    // do all reduce
    MPI_CALL(MPI_Allreduce(MPI_IN_PLACE, recvbuf, count, mpi_dtype, MPI_SUM, context.world_comm));

    // copy back
    if (unit.begin + 1 != unit.end) {
      size_t offset = 0;

      for (size_t i = unit.begin; i < unit.end; ++i) {
        std::memcpy(tasks[i].tensor->ptr(), buffer_.ptr(offset), tasks[i].tensor->num_bytes());

        offset += tasks[i].tensor->num_bytes();
      }
    }

    // callback tensor
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
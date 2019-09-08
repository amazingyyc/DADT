#include "device.h"
#include "mpi_all_reduce_executor.h"

namespace dadt {

MPIAllReduceExecutor::MPIAllReduceExecutor(): buffer_(get_cpu_device()) {
}

std::shared_ptr<LockTensor> MPIAllReduceExecutor::have_midway_tensor(std::string name) {
  if (tensor_pool_.find(name) != tensor_pool_.end()) {
    return tensor_pool_[name];
  }

  return std::shared_ptr<LockTensor>();
}

std::shared_ptr<LockTensor> MPIAllReduceExecutor::create_midway_tensor(std::string name, std::vector<int> dims, ElementType element_type) {
  if (tensor_pool_.find(name) != tensor_pool_.end()) {
    // have created the tensor, resue it
    auto tensor = tensor_pool_[name];

    ARGUMENT_CHECK(tensor->shape() == Shape(dims) && tensor->element_type() == element_type, "get tensor error!");

    return tensor;
  }

  // create a cpu tensor
  auto device = get_cpu_device();

  Shape shape(dims);

  auto storage = TensorStorage::create(device, shape.size() * element_type.byte_width());

  // the tensor in allreduce inited status is waitforfetch
  auto tensor = std::make_shared<LockTensor>(storage, 0, shape, element_type, name, LockTensorStatus::WaitForFetch);
  
  tensor_pool_[name] = tensor;

  return tensor;
}

void MPIAllReduceExecutor::operator()(const Context &context, const std::vector<Task> &tasks) {
  // mpi all reduce only support cpu tensor and float/double
  auto element_type = tasks[0].tensor->element_type();

  ARGUMENT_CHECK(element_type.is<float>() || element_type.is<double>(), "MPIAllReduceExecutor only support float/double");

  for (auto &task : tasks) {
    ARGUMENT_CHECK(DeviceType::CPU == task.tensor->device()->device_type() && 
                  element_type == task.tensor->element_type(), 
                  "mpi all reduce only support cpu tensor, element type must be float/double")
  }

  void *recvbuf = nullptr;
  int count = 0;

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
      std::memcpy(buffer_.ptr(offset), task.tensor->ptr(), task.tensor->num_bytes());

      offset += task.tensor->num_bytes();
    }

    recvbuf = buffer_.ptr();
  } else {
    recvbuf = tasks[0].tensor->ptr();
    count   = tasks[0].tensor->size();
  }

  auto mpi_dtype = mpi_data_type(context, tasks[0].tensor->element_type());

  // do all reduce
  MPI_CALL(MPI_Allreduce(MPI_IN_PLACE, recvbuf, count, mpi_dtype, MPI_SUM, context.world_comm));

  // copy back to tensor
  if (tasks.size() > 1) {
    size_t offset = 0;

    for (auto &task : tasks) {
      std::memcpy(task.tensor->ptr(), buffer_.ptr(offset), task.tensor->num_bytes());
      
      offset += task.tensor->num_bytes();
    }
  }
}


}
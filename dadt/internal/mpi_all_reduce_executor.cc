#include "device.h"
#include "mpi_all_reduce_executor.h"

namespace dadt {

MPIAllReduceExecutor::MPIAllReduceExecutor(): buffer_(get_cpu_device()) {
}

// if has already create a midway tensor
std::shared_ptr<LockTensor> MPIAllReduceExecutor::has_midway_tensor(std::string name) {
  if (tensor_pool_.find(name) != tensor_pool_.end()) {
    return tensor_pool_[name];
  }

  return std::shared_ptr<LockTensor>();
}


std::shared_ptr<LockTensor> MPIAllReduceExecutor::midway_tensor(std::string name, std::vector<int> dims, ElementType element_type) {
  if (tensor_pool_.find(name) != tensor_pool_.end()) {
    // have created the tensor, resue it
    auto tensor = tensor_pool_[name];

    ARGUMENT_CHECK(tensor->shape() == Shape(dims) && tensor->element_type() == element_type, "get tesnor error!");

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
  void *recvbuf = nullptr;
  int count = 0;

  if (tasks.size() > 1 || DeviceType::CPU != tasks[0].tensor->device()->device_type()) {
    size_t memory_size = 0;

    for (auto &t : tasks) {
      count += t.tensor->size();
      memory_size += t.tensor->num_bytes();
    }

    // reserve enough memory
    buffer_.reserve(memory_size);

    // copy tensor to buffer
    size_t offset = 0;

    for (auto &t : tasks) {
      t.tensor->copy_to_cpu(buffer_.ptr(offset));

      offset += t.tensor->num_bytes();
    }

    recvbuf = buffer_.ptr();
  } else {
    recvbuf = tasks[0].tensor->ptr();
    count   = tasks[0].tensor->size();
  }

  // do all reduce
  MPI_CALL(MPI_Allreduce(MPI_IN_PLACE, recvbuf, count, MPI_FLOAT, MPI_SUM, context.world_comm));

  // copy back to tensor
  if (tasks.size() > 1 || DeviceType::CPU != tasks[0].tensor->device()->device_type()) {
    size_t offset = 0;

    for (auto &t : tasks) {
      t.tensor->copy_from_cpu(buffer_.ptr(offset));

      offset += t.tensor->num_bytes();
    }
  }
}


}
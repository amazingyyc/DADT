#include "mpi_broad_cast_executor.h"

namespace dadt {

MPIBroadCastExecutor::MPIBroadCastExecutor(): buffer_(get_cpu_device()) {
}

// if has already create a midway tensor
std::shared_ptr<LockTensor> MPIBroadCastExecutor::has_midway_tensor(std::string name) {
  return std::shared_ptr<LockTensor>();
}

std::shared_ptr<LockTensor> MPIBroadCastExecutor::midway_tensor(std::string name, std::vector<int> dims, ElementType element_type) {
  // the broadcast executor only works once, so we do not store the tensor for reuse
  // MPI broadcast should need cpu tensor
  auto device = get_cpu_device();

  // tensor shape
  Shape shape(dims);

  // create a tensor storage
  auto storage = TensorStorage::create(device, shape.size() * element_type.byte_width());

  // broadcast tensor inited status is InFill
  // the status is not work for broadcast
  auto tensor = std::make_shared<LockTensor>(storage, 0, shape, element_type, name, LockTensorStatus::InFill);

  return tensor;
}

void MPIBroadCastExecutor::operator()(const Context &context, const std::vector<Task> &tasks) {
  void *sendbuf = nullptr;
  int count = 0;

  // if the tensor count > 1 or it is a GPU tensor, copy to memorybuffer
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

    sendbuf = buffer_.ptr();
  } else {
    sendbuf = tasks[0].tensor->ptr();
    count   = tasks[0].tensor->size();
  }

  // broad cast
  MPI_CALL(MPI_Bcast(sendbuf, count, MPI_FLOAT, 0, context.world_comm));

  if (tasks.size() > 1 || DeviceType::CPU != tasks[0].tensor->device()->device_type()) {
    size_t offset = 0;

    for (auto &t : tasks) {
      t.tensor->copy_from_cpu(buffer_.ptr(offset));

      offset += t.tensor->num_bytes();
    }
  }
}

}
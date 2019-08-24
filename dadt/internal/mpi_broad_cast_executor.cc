#include "mpi_broad_cast_executor.h"

namespace dadt {

MPIBroadCastExecutor::MPIBroadCastExecutor() {
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
  // for broad cast we will broad one by one
  for (auto &task : tasks) {
    ARGUMENT_CHECK(DeviceType::CPU == task.tensor->device()->device_type(), "MPIBroadCastExecutor only support CPU tensor");

    void *sendbuf = task.tensor->ptr();;
    int count     = task.tensor->size();

    auto mpi_dtype = mpi_data_type(context, task.tensor->element_type());

    // broad cast from rank 0
    MPI_CALL(MPI_Bcast(sendbuf, count, mpi_dtype, 0, context.world_comm));
  }
}

}
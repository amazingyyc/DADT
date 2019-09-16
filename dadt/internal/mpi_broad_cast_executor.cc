#include "mpi_broad_cast_executor.h"

namespace dadt {

MPIBroadCastExecutor::MPIBroadCastExecutor() {
}

std::shared_ptr<LockTensor> MPIBroadCastExecutor::obtain_midway_tensor(std::string name) {
  return std::shared_ptr<LockTensor>();
}

std::shared_ptr<LockTensor> MPIBroadCastExecutor::create_midway_tensor(std::string name, std::vector<int> dims, ElementType element_type) {
  // the broadcast executor only works once, so we do not store the tensor for reuse
  // MPI broadcast should need cpu tensor
  auto device = get_cpu_device();

  // tensor shape
  Shape shape(dims);

  // create a tensor storage
  auto storage = TensorStorage::create(device, shape.size() * element_type.byte_width());

  // broadcast tensor inited status is InFetch
  // the status is not work for broadcast
  auto tensor = std::make_shared<LockTensor>(storage, 0, shape, element_type, name, LockTensorStatus::InFetch);

  return tensor;
}

void MPIBroadCastExecutor::operator()(const Context &context, const std::vector<Task> &tasks, std::shared_ptr<TimeLine> timeline) {
  // begin broad cast timeline
  if (context.enable_timeline.load()) {
    timeline->begin(tasks, kDoBroadCastEvent);
  }

  // for broad cast we will broad one by one
  for (auto &task : tasks) {
    ARGUMENT_CHECK(DeviceType::CPU == task.tensor->device()->device_type(), "MPIBroadCastExecutor only support CPU tensor");

    void *sendbuf = task.tensor->ptr();
    int count = task.tensor->size();

    auto mpi_dtype = mpi_data_type(context, task.tensor->element_type());

    // broad cast from rank 0
    MPI_CALL(MPI_Bcast(sendbuf, count, mpi_dtype, 0, context.world_comm));
  }

  for (auto &task : tasks) {
    task.done();
  }

  // end broad cast timeline
  if (context.enable_timeline.load()) {
    timeline->end(tasks, kDoBroadCastEvent);
  }
}

}
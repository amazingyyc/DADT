#include "mpi_broad_cast_executor.h"

namespace dadt {

MPIBroadCastExecutor::MPIBroadCastExecutor(std::shared_ptr<Device> cpu_device): cpu_device_(cpu_device) {
}

std::shared_ptr<LockTensor> MPIBroadCastExecutor::obtain_midway_tensor(std::string name) {
  return nullptr;
}

std::shared_ptr<LockTensor> MPIBroadCastExecutor::create_midway_tensor(std::string name, std::vector<int> dims, ElementType element_type) {
  // tensor shape
  Shape shape(dims);

  // MPI broadcast should need cpu tensor
  auto storage = TensorStorage::create(cpu_device_, shape.size() * element_type.byte_width());

  // broadcast tensor inited status is kInFetch
  // the status is not work for broadcast
  auto tensor = std::make_shared<LockTensor>(storage, 0, shape, element_type, name, LockTensorStatus::kInFetch);

  return tensor;
}

void MPIBroadCastExecutor::operator()(const Context &context, const std::vector<Task> &tasks, std::shared_ptr<TimeLine> timeline) {
  // begin broad cast timeline
  // if (context.enable_timeline.load()) {
  //   timeline->begin(tasks, kDoBroadCastEvent);
  // }

  // for broad cast we will broad one by one
  for (auto &task : tasks) {
    ARGUMENT_CHECK(DeviceType::CPU == task.tensor->device()->device_type(), "MPIBroadCastExecutor only support CPU tensor");

    void *sendbuf = task.tensor->ptr();
    int count = task.tensor->size();

    auto mpi_dtype = mpi_data_type(context, task.tensor->element_type());

    // broad cast from rank 0
    MPI_CALL(MPI_Bcast(sendbuf, count, mpi_dtype, 0, context.world_comm));

    // finish callback
    task.done();

    // if (context.enable_timeline.load()) {
    //   timeline->end(task.name, kDoBroadCastEvent);
    // }
  }
}

}
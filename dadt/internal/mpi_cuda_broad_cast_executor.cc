#include "mpi_cuda_broad_cast_executor.h"

namespace dadt {

MPICUDABroadCastExecutor::MPICUDABroadCastExecutor(std::shared_ptr<Device> gpu_device): gpu_device_(gpu_device) {
}

MPICUDABroadCastExecutor::~MPICUDABroadCastExecutor() {
}

std::shared_ptr<LockTensor> MPICUDABroadCastExecutor::obtain_midway_tensor(std::string name) {
  return std::shared_ptr<LockTensor>();
}

std::shared_ptr<LockTensor> MPICUDABroadCastExecutor::create_midway_tensor(std::string name, std::vector<int> dims, ElementType element_type) {
  Shape shape(dims);

  auto storage = TensorStorage::create(gpu_device_, shape.size() * element_type.byte_width());

  auto tensor = std::make_shared<LockTensor>(storage, 0, shape, element_type, name, LockTensorStatus::kInFetch);

  return tensor;
}

void MPICUDABroadCastExecutor::operator()(const Context &context, const std::vector<Task> &tasks, std::shared_ptr<TimeLine> timeline) {
  if (context.enable_timeline.load()) {
    timeline->begin(tasks, kDoBroadCastEvent);
  }

  for (auto &task : tasks) {
    ARGUMENT_CHECK(DeviceType::GPU == task.tensor->device()->device_type(), "MPICUDABroadCastExecutor only support GPU tensor");

    void *sendbuf = task.tensor->ptr();
    int count = task.tensor->size();

    auto mpi_dtype = mpi_data_type(context, task.tensor->element_type());

    // broad cast from rank 0
    MPI_CALL(MPI_Bcast(sendbuf, count, mpi_dtype, 0, context.world_comm));

    // finish callback
    task.done();

    if (context.enable_timeline.load()) {
      timeline->end(task.name, kDoBroadCastEvent);
    }
  }


}

}


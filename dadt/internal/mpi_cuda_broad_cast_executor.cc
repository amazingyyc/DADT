#include "mpi_cuda_broad_cast_executor.h"

namespace dadt {

MPICUDABroadCastExecutor::MPICUDABroadCastExecutor(std::shared_ptr<Device> gpu_device): gpu_device_(gpu_device) {
}

MPICUDABroadCastExecutor::~MPICUDABroadCastExecutor() {
}

bool MPICUDABroadCastExecutor::is_cuda_midway_tensor() {
  return true;
}

std::shared_ptr<LockTensor> MPICUDABroadCastExecutor::obtain_midway_tensor(std::string name) {
  return nullptr;
}

std::shared_ptr<LockTensor> MPICUDABroadCastExecutor::create_midway_tensor(std::string name, Shape shape, ElementType element_type) {
  auto storage = TensorStorage::create(gpu_device_, shape.size() * element_type.byte_width());

  auto tensor = std::make_shared<LockTensor>(storage, 0, shape, element_type, name, LockTensorStatus::kInFetch);

  return tensor;
}

void MPICUDABroadCastExecutor::operator()(const Context &context, const std::vector<Task> &tasks, std::shared_ptr<TimeLine> timeline) {
  // if (context.enable_timeline.load()) {
  //   timeline->begin(tasks, kDoBroadCastEvent);
  // }

  for (auto &task : tasks) {
    ARGUMENT_CHECK(task.tensor->is_cuda(), "MPICUDABroadCastExecutor only support GPU tensor");

    if (task.before) {
      task.before();
    }

    void *sendbuf = task.tensor->ptr();
    int count = task.tensor->size();

    auto mpi_dtype = mpi_data_type(context, task.tensor->element_type());

    // broad cast from rank 0
    MPI_CALL(MPI_Bcast(sendbuf, count, mpi_dtype, 0, context.world_comm));

    // finish callback
    if (task.done) {
      task.done();
    }

    // if (context.enable_timeline.load()) {
    //   timeline->end(task.name, kDoBroadCastEvent);
    // }
  }
}

}


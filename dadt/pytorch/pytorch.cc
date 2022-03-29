#include "pytorch.h"

#ifdef HAVE_NCCL
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#endif

#include <memory>
#include <mutex>

#include "commander.h"
#include "common/exception.h"
#include "common/task.h"
#include "pytorch_tensor_impl.h"
#include "t/lock_tensor.h"

namespace dadt {
namespace pytorch {

std::once_flag flag;
Commander commander;

void Initialize(const Config& config) {
  std::call_once(flag, [&config]() { commander.Initialize(config); });
}

void Shutdown() {
  commander.Shutdown();
}

bool Initialized() {
  return commander.Initialized();
}

int Size() {
  return commander.Size();
}

int LocalSize() {
  return commander.LocalSize();
}

int Rank() {
  return commander.Rank();
}

int LocalRank() {
  return commander.LocalRank();
}

void Barrier() {
  commander.Barrier();
}

void LocalBarrier() {
  commander.LocalBarrier();
}

torch::Tensor BroadCastCpu(uint32_t id, torch::Tensor input) {
  // Clone it, make sure not modify the origin tensor
  std::shared_ptr<PytorchTensorImpl> impl(new PytorchTensorImpl(input.clone()));
  std::shared_ptr<LockTensor> l_tensor(
      new LockTensor(LockTensorStatus::kInExecute, Tensor(impl)));

  Task task;
  task.type = kBroadCastTaskType;
  task.id = id;
  task.l_tensor = l_tensor;
  task.done = [l_tensor]() {
    // When finish, change the status.
    l_tensor->Wait(LockTensorStatus::kInExecute,
                   LockTensorStatus::kWaitForFetch);
  };

  commander.EnqueueTask(std::move(task));

  l_tensor->Wait(LockTensorStatus::kWaitForFetch, LockTensorStatus::kInFetch);

  return impl->torch_tensor();
}

#ifdef HAVE_NCCL
torch::Tensor BroadCastGpu(uint32_t id, torch::Tensor input) {
  cudaStream_t cuda_stream =
      c10::cuda::getCurrentCUDAStream(input.device().index()).stream();

  std::shared_ptr<PytorchTensorImpl> impl(new PytorchTensorImpl(input.clone()));
  std::shared_ptr<LockTensor> l_tensor(
      new LockTensor(LockTensorStatus::kInExecute, Tensor(impl)));

  cudaEvent_t cuda_event = impl->cuda_event();

  // Put a event into CudaStream.
  // The GPU is async so for now the input's data maybe not ready.
  CUDA_CALL(cudaEventRecord(cuda_event, cuda_stream));

  Task task;
  task.type = kBroadCastTaskType;
  task.id = id;
  task.l_tensor = l_tensor;
  task.before = [cuda_event]() {
    // Wait clone ready.
    CUDA_CALL(cudaEventSynchronize(cuda_event));
  };

  task.done = [l_tensor]() {
    // When finish, change the status.
    l_tensor->Wait(LockTensorStatus::kInExecute,
                   LockTensorStatus::kWaitForFetch);
  };

  commander.EnqueueTask(std::move(task));

  l_tensor->Wait(LockTensorStatus::kWaitForFetch, LockTensorStatus::kInFetch);

  return impl->torch_tensor();
}
#endif

torch::Tensor BroadCast(uint32_t id, torch::Tensor input) {
  if (input.is_cuda()) {
#ifdef HAVE_NCCL
    return BroadCastGpu(id, input);
#else
    RUNTIME_ERROR("DADT does't build with GPU")
#endif
  } else {
    return BroadCastCpu(id, input);
  }
}

torch::Tensor AllReduce(uint32_t id, torch::Tensor input) {
  // Clone it, make sure not modify the origin tensor
  std::shared_ptr<PytorchTensorImpl> impl(new PytorchTensorImpl(input.clone()));

  std::shared_ptr<LockTensor> l_tensor(
      new LockTensor(LockTensorStatus::kInExecute, Tensor(impl)));

  Task task;
  task.type = kAllReduceTaskType;
  task.id = id;
  task.l_tensor = l_tensor;
  task.done = [l_tensor]() {
    // When finish, change the status.
    l_tensor->Wait(LockTensorStatus::kInExecute,
                   LockTensorStatus::kWaitForFetch);
  };

  commander.EnqueueTask(std::move(task));

  l_tensor->Wait(LockTensorStatus::kWaitForFetch, LockTensorStatus::kInFetch);

  return dynamic_cast<PytorchTensorImpl*>(l_tensor->tensor().impl().get())
      ->torch_tensor();
}

torch::Tensor AllReduceAsync(uint32_t id, torch::Tensor input) {
  TaskKey key;
  key.type = kAllReduceTaskType;
  key.id = id;

  std::shared_ptr<LockTensor> l_tensor = commander.CachedLTensor(key);
  if (l_tensor == nullptr) {
    // Firstly time.
    auto options = torch::TensorOptions()
                       .dtype(input.dtype())
                       .layout(input.layout())
                       .device(input.device());

    auto zeros = torch::zeros_like(input, options);

    std::shared_ptr<PytorchTensorImpl> zeros_impl(new PytorchTensorImpl(zeros));

    l_tensor = std::shared_ptr<LockTensor>(
        new LockTensor(LockTensorStatus::kWaitForFetch, Tensor(zeros_impl)));

    commander.InsertLTensor(key, l_tensor);
  }

  // Wait finish AllReduce.
  l_tensor->Wait(LockTensorStatus::kWaitForFetch, LockTensorStatus::kInFetch);

  torch::Tensor output;
  {
    PytorchTensorImpl* pytorch_impl =
        dynamic_cast<PytorchTensorImpl*>(l_tensor->tensor().impl().get());

    output = pytorch_impl->torch_tensor();
  }

  // Put input in.
  l_tensor->ResetTensor(Tensor(std::make_shared<PytorchTensorImpl>(input)));

  Task task;
  task.type = kAllReduceTaskType;
  task.id = id;
  task.l_tensor = l_tensor;
  task.done = [l_tensor]() {
    l_tensor->Wait(LockTensorStatus::kInExecute,
                   LockTensorStatus::kWaitForFetch);
  };

  l_tensor->Wait(dadt::LockTensorStatus::kInFetch,
                 dadt::LockTensorStatus::kInExecute);

  commander.EnqueueTask(std::move(task));

  return output;
}

torch::Tensor CooAllReduce(uint32_t id, torch::Tensor input) {
  // Coo AllReduce no need clone, CooAllReduce will not modify the origin
  // tensor.
  ARGUMENT_CHECK(input.is_sparse(), "CooAllReduce need input is Coo format");

  std::shared_ptr<PytorchTensorImpl> impl(new PytorchTensorImpl(input));

  std::shared_ptr<LockTensor> l_tensor(
      new LockTensor(LockTensorStatus::kInExecute, Tensor(impl)));

  Task task;
  task.type = kCooAllReduceTaskType;
  task.id = id;
  task.l_tensor = l_tensor;
  task.done = [l_tensor]() {
    // When finish, change the status.
    l_tensor->Wait(LockTensorStatus::kInExecute,
                   LockTensorStatus::kWaitForFetch);
  };

  commander.EnqueueTask(std::move(task));

  l_tensor->Wait(LockTensorStatus::kWaitForFetch, LockTensorStatus::kInFetch);

  return dynamic_cast<PytorchTensorImpl*>(l_tensor->tensor().impl().get())
      ->torch_tensor();
}

torch::Tensor CooAllReduceAsync(uint32_t id, torch::Tensor input) {
  ARGUMENT_CHECK(input.is_sparse(), "CooAllReduce need input is Coo format");

  TaskKey key;
  key.type = kAllReduceTaskType;
  key.id = id;

  std::shared_ptr<LockTensor> l_tensor = commander.CachedLTensor(key);
  if (l_tensor == nullptr) {
    auto values_options = torch::TensorOptions()
                              .dtype(input.values().dtype())
                              .layout(input.values().layout())
                              .device(input.values().device());

    auto zero_values = torch::zeros_like(input.values(), values_options);

    torch::Tensor zeros = torch::sparse_coo_tensor(input.indices().clone(),
                                                   zero_values, input.sizes());

    l_tensor = std::shared_ptr<LockTensor>(
        new LockTensor(LockTensorStatus::kWaitForFetch,
                       Tensor(std::make_shared<PytorchTensorImpl>(zeros))));

    commander.InsertLTensor(key, l_tensor);
  }

  // Wait finish AllReduce.
  l_tensor->Wait(LockTensorStatus::kWaitForFetch, LockTensorStatus::kInFetch);

  PytorchTensorImpl* pytorch_impl =
      dynamic_cast<PytorchTensorImpl*>(l_tensor->tensor().impl().get());

  torch::Tensor output = pytorch_impl->torch_tensor();

  // Put input in.
  // (TODO) At here must clone input or will "hange" when call ResetTensor. I
  // donot know why.
  l_tensor->ResetTensor(
      Tensor(std::make_shared<PytorchTensorImpl>(input.clone())));

  Task task;
  task.type = kCooAllReduceTaskType;
  task.id = id;
  task.l_tensor = l_tensor;
  task.done = [l_tensor]() {
    l_tensor->Wait(LockTensorStatus::kInExecute,
                   LockTensorStatus::kWaitForFetch);
  };

  l_tensor->Wait(dadt::LockTensorStatus::kInFetch,
                 dadt::LockTensorStatus::kInExecute);

  commander.EnqueueTask(std::move(task));

  return output;
}

}  // namespace pytorch
}  // namespace dadt

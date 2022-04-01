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
#include "pytorch_utils.h"
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
  auto impl = std::make_shared<PytorchTensorImpl>(input.clone());
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
      at::cuda::getCurrentCUDAStream(input.device().index()).stream();

  auto impl = std::make_shared<PytorchTensorImpl>(input.clone());
  std::shared_ptr<LockTensor> l_tensor(
      new LockTensor(LockTensorStatus::kInExecute, Tensor(impl)));

  // Put a event into pytorch's CudaStream.
  // The GPU is async so for now the input's data maybe not ready.
  CUDA_CALL(cudaEventRecord(l_tensor->cuda_event(), cuda_stream));

  Task task;
  task.type = kBroadCastTaskType;
  task.id = id;
  task.l_tensor = l_tensor;
  task.before = [l_tensor]() {
    // Wait clone ready.
    l_tensor->CudaEventSynchronize();
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

torch::Tensor AllReduceCpu(uint32_t id, torch::Tensor input) {
  // Clone it, make sure not modify the origin tensor
  auto impl = std::make_shared<PytorchTensorImpl>(input.clone());
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

  return impl->torch_tensor();
}

#ifdef HAVE_NCCL
torch::Tensor AllReduceGpu(uint32_t id, torch::Tensor input) {
  cudaStream_t cuda_stream =
      at::cuda::getCurrentCUDAStream(input.device().index()).stream();

  auto impl = std::make_shared<PytorchTensorImpl>(input.clone());
  std::shared_ptr<LockTensor> l_tensor(
      new LockTensor(LockTensorStatus::kInExecute, Tensor(impl)));

  // Put a event into pytorch's CudaStream.
  // The GPU is async so for now the input's data maybe not ready.
  CUDA_CALL(cudaEventRecord(l_tensor->cuda_event(), cuda_stream));

  Task task;
  task.type = kAllReduceTaskType;
  task.id = id;
  task.l_tensor = l_tensor;
  task.before = [l_tensor]() {
    // Wait clone ready.
    l_tensor->CudaEventSynchronize();
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

torch::Tensor AllReduce(uint32_t id, torch::Tensor input) {
  if (input.is_cuda()) {
#ifdef HAVE_NCCL
    return AllReduceGpu(id, input);
#else
    RUNTIME_ERROR("DADT does't build with GPU")
#endif
  } else {
    return AllReduceCpu(id, input);
  }
}

torch::Tensor AllReduceAsyncCpu(uint32_t id, torch::Tensor input) {
  TaskKey key;
  key.type = kAllReduceTaskType;
  key.id = id;

  torch::Tensor output;

  std::shared_ptr<LockTensor> l_tensor = commander.CachedLTensor(key);
  if (l_tensor == nullptr) {
    // First time.
    output = input.clone().zero_();

    auto impl = std::make_shared<PytorchTensorImpl>(input);
    l_tensor = std::shared_ptr<LockTensor>(
        new LockTensor(LockTensorStatus::kInFetch, Tensor(impl)));

    commander.InsertLTensor(key, l_tensor);
  } else {
    // Wait finish AllReduce.
    l_tensor->Wait(LockTensorStatus::kWaitForFetch, LockTensorStatus::kInFetch);

    PytorchTensorImpl* pytorch_impl =
        dynamic_cast<PytorchTensorImpl*>(l_tensor->tensor().impl().get());

    output = pytorch_impl->torch_tensor();

    // Put input in.
    l_tensor->ResetTensor(Tensor(std::make_shared<PytorchTensorImpl>(input)));
  }

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

#ifdef HAVE_NCCL
torch::Tensor AllReduceAsyncGpu(uint32_t id, torch::Tensor input) {
  cudaStream_t cuda_stream =
      at::cuda::getCurrentCUDAStream(input.device().index()).stream();

  TaskKey key;
  key.type = kAllReduceTaskType;
  key.id = id;

  torch::Tensor output;

  std::shared_ptr<LockTensor> l_tensor = commander.CachedLTensor(key);
  if (l_tensor == nullptr) {
    // First time.
    output = input.clone().zero_();

    auto impl = std::make_shared<PytorchTensorImpl>(input);
    l_tensor = std::shared_ptr<LockTensor>(
        new LockTensor(LockTensorStatus::kInFetch, Tensor(impl)));

    commander.InsertLTensor(key, l_tensor);
  } else {
    l_tensor->Wait(dadt::LockTensorStatus::kWaitForFetch,
                   dadt::LockTensorStatus::kInFetch);

    PytorchTensorImpl* impl =
        dynamic_cast<PytorchTensorImpl*>(l_tensor->tensor().impl().get());

    output = impl->torch_tensor();

    // Put current input into the LockTensor.
    l_tensor->ResetTensor(Tensor(std::make_shared<PytorchTensorImpl>(input)));
  }

  // Put a event into pytorch's CudaStream.
  // The GPU is async so for now the input's data maybe not ready.
  CUDA_CALL(cudaEventRecord(l_tensor->cuda_event(), cuda_stream));

  Task task;
  task.type = kAllReduceTaskType;
  task.id = id;
  task.l_tensor = l_tensor;
  task.before = [l_tensor]() { l_tensor->CudaEventSynchronize(); };
  task.done = [l_tensor]() {
    l_tensor->Wait(LockTensorStatus::kInExecute,
                   LockTensorStatus::kWaitForFetch);
  };

  l_tensor->Wait(dadt::LockTensorStatus::kInFetch,
                 dadt::LockTensorStatus::kInExecute);

  commander.EnqueueTask(std::move(task));

  return output;
}
#endif

torch::Tensor AllReduceAsync(uint32_t id, torch::Tensor input) {
  if (input.is_cuda()) {
#ifdef HAVE_NCCL
    return AllReduceAsyncGpu(id, input);
#else
    RUNTIME_ERROR("DADT does't build with GPU")
#endif
  } else {
    return AllReduceAsyncCpu(id, input);
  }
}

torch::Tensor CooAllReduceCpu(uint32_t id, torch::Tensor input) {
  ARGUMENT_CHECK(input.is_sparse(), "CooAllReduce need input is Coo format");

  if (input.is_coalesced() == false) {
    input = input.coalesce();
  }

  std::shared_ptr<LockTensor> l_tensor(
      new LockTensor(LockTensorStatus::kInExecute, CooTensorFromTorch(input)));

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

  return CooTensorToTorch(l_tensor->tensor());
}

#ifdef HAVE_NCCL
torch::Tensor CooAllReduceGpu(uint32_t id, torch::Tensor input) {
  ARGUMENT_CHECK(input.is_sparse(), "CooAllReduce need input is Coo format");

  if (input.is_coalesced() == false) {
    input = input.coalesce();
  }

  cudaStream_t cuda_stream =
      at::cuda::getCurrentCUDAStream(input.device().index()).stream();

  std::shared_ptr<LockTensor> l_tensor(
      new LockTensor(LockTensorStatus::kInExecute, CooTensorFromTorch(input)));

  // Put a event into pytorch's CudaStream.
  // The GPU is async so for now the input's data maybe not ready.
  CUDA_CALL(cudaEventRecord(l_tensor->cuda_event(), cuda_stream));

  Task task;
  task.type = kCooAllReduceTaskType;
  task.id = id;
  task.l_tensor = l_tensor;
  task.before = [l_tensor]() { l_tensor->CudaEventSynchronize(); };

  task.done = [l_tensor]() {
    // When finish, change the status.
    l_tensor->Wait(LockTensorStatus::kInExecute,
                   LockTensorStatus::kWaitForFetch);
  };

  commander.EnqueueTask(std::move(task));

  l_tensor->Wait(LockTensorStatus::kWaitForFetch, LockTensorStatus::kInFetch);

  return CooTensorToTorch(l_tensor->tensor());
}
#endif

torch::Tensor CooAllReduce(uint32_t id, torch::Tensor input) {
  if (input.is_cuda()) {
#ifdef HAVE_NCCL
    return CooAllReduceGpu(id, input);
#else
    RUNTIME_ERROR("DADT does't build with GPU")
#endif
  } else {
    return CooAllReduceCpu(id, input);
  }
}

torch::Tensor CooAllReduceAsyncCpu(uint32_t id, torch::Tensor input) {
  ARGUMENT_CHECK(input.is_sparse(), "CooAllReduce need input is Coo format");

  if (input.is_coalesced() == false) {
    input = input.coalesce();
  }

  TaskKey key;
  key.type = kCooAllReduceTaskType;
  key.id = id;

  torch::Tensor output;

  std::shared_ptr<LockTensor> l_tensor = commander.CachedLTensor(key);
  if (l_tensor == nullptr) {
    // Clone indices and values avoid call input.clone().zero().
    // input.clone().zero() will return a new coo tensor with different shape.
    output = torch::sparse_coo_tensor(
        input.indices().clone(), input.values().clone().zero_(), input.sizes());

    // At here must clone the input or will be "hang" when release the pytorch
    // tensor. I do't know why.
    l_tensor = std::shared_ptr<LockTensor>(
        new LockTensor(LockTensorStatus::kInFetch, CooTensorFromTorch(input)));

    commander.InsertLTensor(key, l_tensor);
  } else {
    // Wait finish AllReduce.
    l_tensor->Wait(LockTensorStatus::kWaitForFetch, LockTensorStatus::kInFetch);

    PytorchTensorImpl* impl =
        dynamic_cast<PytorchTensorImpl*>(l_tensor->tensor().impl().get());

    output = CooTensorToTorch(l_tensor->tensor());

    // Put current input into the LockTensor.
    // At here must clone the input or will be "hang" when release the pytorch
    // tensor. I do't know why.
    l_tensor->ResetTensor(CooTensorFromTorch(input));
  }

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

#ifdef HAVE_NCCL
torch::Tensor CooAllReduceAsyncGpu(uint32_t id, torch::Tensor input) {
  ARGUMENT_CHECK(input.is_sparse(), "CooAllReduce need input is Coo format");

  if (input.is_coalesced() == false) {
    input = input.coalesce();
  }

  cudaStream_t cuda_stream =
      at::cuda::getCurrentCUDAStream(input.device().index()).stream();

  TaskKey key;
  key.type = kCooAllReduceTaskType;
  key.id = id;

  torch::Tensor output;

  std::shared_ptr<LockTensor> l_tensor = commander.CachedLTensor(key);
  if (l_tensor == nullptr) {
    // Clone indices and values avoid call input.clone().zero().
    // input.clone().zero() will return a new coo tensor with different shape.
    output = torch::sparse_coo_tensor(
        input.indices().clone(), input.values().clone().zero_(), input.sizes());

    l_tensor = std::shared_ptr<LockTensor>(
        new LockTensor(LockTensorStatus::kInFetch, CooTensorFromTorch(input)));

    commander.InsertLTensor(key, l_tensor);
  } else {
    l_tensor->Wait(dadt::LockTensorStatus::kWaitForFetch,
                   dadt::LockTensorStatus::kInFetch);

    output = CooTensorToTorch(l_tensor->tensor());

    // Put current input into the LockTensor.
    l_tensor->ResetTensor(CooTensorFromTorch(input));
  }

  // Put a event into pytorch's CudaStream.
  // The GPU is async so for now the input's data maybe not ready.
  CUDA_CALL(cudaEventRecord(l_tensor->cuda_event(), cuda_stream));

  Task task;
  task.type = kCooAllReduceTaskType;
  task.id = id;
  task.l_tensor = l_tensor;
  task.before = [l_tensor]() { l_tensor->CudaEventSynchronize(); };
  task.done = [l_tensor]() {
    l_tensor->Wait(LockTensorStatus::kInExecute,
                   LockTensorStatus::kWaitForFetch);
  };

  l_tensor->Wait(dadt::LockTensorStatus::kInFetch,
                 dadt::LockTensorStatus::kInExecute);

  commander.EnqueueTask(std::move(task));

  return output;
}
#endif

torch::Tensor CooAllReduceAsync(uint32_t id, torch::Tensor input) {
  if (input.is_cuda()) {
#ifdef HAVE_NCCL
    return CooAllReduceAsyncGpu(id, input);
#else
    RUNTIME_ERROR("DADT does't build with GPU")
#endif
  } else {
    return CooAllReduceAsyncCpu(id, input);
  }
}

}  // namespace pytorch
}  // namespace dadt

#pragma once

#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <nccl.h>
#include <torch/extension.h>
#include <torch/torch.h>

#include "common/stream_guard.h"

namespace dadt {
namespace pytorch {

// ref 201 line
// in:https://github.com/pytorch/pytorch/blob/master/c10/cuda/CUDAStream.cpp
// ref:https://pytorch.org/cppdocs/api/function_namespacec10_1_1cuda_1a399b2870dd4f6e4f9424666762080332.html
class PytorchCudaStreamGuard : public StreamGuard {
private:
  // The Pytorch implement of CUDAStreamGuard is based on ThreadLocal, ref
  // line(136):https://github.com/pytorch/pytorch/blob/master/c10/cuda/CUDAStream.cpp
  // So It's threadsafe to call this guard in other thread.
  at::cuda::CUDAStreamGuard guard_;

public:
  PytorchCudaStreamGuard(cudaStream_t cuda_stream, int8_t device_index)
      : guard_(at::cuda::getStreamFromExternal(cuda_stream, device_index)) {
  }

  ~PytorchCudaStreamGuard() = default;
};

}  // namespace pytorch
}  // namespace dadt

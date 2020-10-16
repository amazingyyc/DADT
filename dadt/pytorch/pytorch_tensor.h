#ifndef PYTORCH_TENSOR_H
#define PYTORCH_TENSOR_H

#include <torch/extension.h>
#include <torch/torch.h>

#include "lock_tensor.h"

namespace dadt {
namespace pytorch {

class PytorchTensor: public LockTensor {
private:
  // PytorchTensor does not has a TensorStorage it's torch::Tensor wrapper
  torch::Tensor torch_tensor_;

#ifdef HAVE_NCCL
  // if build for cuda every PytorchTensor need a CudaEvent when call beforecallback.
  cudaEvent_t cuda_event_;
#endif

public:
  PytorchTensor(torch::Tensor torch_tensor, std::string name, LockTensorStatus initialize_status);

  ~PytorchTensor();

#ifdef HAVE_NCCL
  cudaEvent_t cuda_event();
#endif

  int device_id() const override;

  bool is_cuda() const override;

  // get memory pointer
  void* ptr() override;
  void* ptr() const override;

  torch::Tensor torch_tensor();
  void torch_tensor(torch::Tensor tensor);
};

}
}

#endif
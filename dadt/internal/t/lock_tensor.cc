#include "lock_tensor.h"

#include "common/exception.h"

namespace dadt {
LockTensor::LockTensor(LockTensorStatus init_status, Tensor tensor)
    : status_((int32_t)init_status), tensor_(tensor) {
  if (tensor_.IsCuda()) {
#ifdef HAVE_NCCL
    CUDA_CALL(cudaEventCreate(&cuda_event_));
#else
    RUNTIME_ERROR("DADT build without CUDA but got a CUDA tensor");
#endif
  }
}

LockTensor::~LockTensor() {
  if (tensor_.IsCuda()) {
#ifdef HAVE_NCCL
    cudaEventDestroy(cuda_event_);
#else
    RUNTIME_ERROR("DADT build without CUDA but got a CUDA tensor");
#endif
  }
}

const Tensor& LockTensor::tensor() const {
  return tensor_;
}

#ifdef HAVE_NCCL
cudaEvent_t LockTensor::cuda_event() const {
  return cuda_event_;
}

void LockTensor::CudaEventSynchronize() const {
  CUDA_CALL(cudaEventSynchronize(cuda_event_));
}

#endif

void LockTensor::ResetTensor(const Tensor& tensor) {
  tensor_ = tensor;
}

void LockTensor::Wait(LockTensorStatus expected_status,
                      LockTensorStatus new_status) {
  status_.Wait((int32_t)expected_status, (int32_t)new_status);
}

}  // namespace dadt

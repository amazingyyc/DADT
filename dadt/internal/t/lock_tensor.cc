#include "lock_tensor.h"

#include "common/exception.h"

namespace dadt {
LockTensor::LockTensor(LockTensorStatus init_status, Tensor tensor)
    : status_((int32_t)init_status), tensor_(tensor) {
}

// wait the tensor is expected_status/new_status and change it to new_status
void LockTensor::Wait(LockTensorStatus expected_status,
                      LockTensorStatus new_status) {
  status_.Wait((int32_t)expected_status, (int32_t)new_status);
}

const Tensor& LockTensor::tensor() const {
  return tensor_;
}

void LockTensor::ResetTensor(const Tensor& tensor) {
  tensor_ = tensor;
}

}  // namespace dadt

#include <iostream>
#include <string>

#include "definition.h"
#include "lock_tensor.h"

namespace dadt {

LockTensor::LockTensor(std::shared_ptr<TensorStorage> storage, 
                      size_t offset, 
                      Shape shape, 
                      ElementType type,
                      std::string name, 
                      LockTensorStatus initialize_status)
  :Tensor(storage, offset, shape, type), name_(name), status_((int)initialize_status) {
}

std::string LockTensor::name() {
  return name_;
}

// wait the tensor is expected_status/new_status and change it to new_status
void LockTensor::wait(LockTensorStatus expected_status, LockTensorStatus new_status) {
  status_.wait((int)expected_status, (int)new_status);
}


}
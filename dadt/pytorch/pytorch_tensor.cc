#include "pytorch_utils.h"
#include "pytorch_tensor.h"

namespace dadt {
namespace pytorch {

PytorchTensor::PytorchTensor(torch::Tensor torch_tensor, std::string name, LockTensorStatus initialize_status)
  :LockTensor(nullptr, 0, dadt::Shape(parse_shape_vector(torch_tensor)), parse_element_type(torch_tensor), name, initialize_status),
   torch_tensor_(torch_tensor) {
}

int PytorchTensor::device_id() const {
  if (torch_tensor_.is_cuda()) {
    return torch_tensor_.device().index();
  }

  return -1;
}

bool PytorchTensor::is_cuda() const {
  return torch_tensor_.is_cuda();
}

void* PytorchTensor::ptr() {
  return torch_tensor_.data_ptr();
}

void* PytorchTensor::ptr() const {
  return torch_tensor_.data_ptr();
}

torch::Tensor PytorchTensor::torch_tensor() {
  return torch_tensor_;
}

void PytorchTensor::torch_tensor(torch::Tensor tensor) {
  torch_tensor_ = tensor;

  // update shape element_type
  shape_ = dadt::Shape(parse_shape_vector(tensor));
  element_type_ = parse_element_type(tensor);
}

}
}
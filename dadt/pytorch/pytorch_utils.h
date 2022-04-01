#pragma once

#include <torch/extension.h>
#include <torch/torch.h>

#include "t/element_type.h"
#include "t/shape.h"
#include "t/tensor.h"

namespace dadt {
namespace pytorch {

ElementType TorchDTypeToElementType(torch::Dtype dtype);

torch::Dtype ElementTypeToTorchDType(ElementType etype);

Shape TorchSizesToShape(const torch::IntArrayRef& sizes);

Tensor CooTensorFromTorch(const torch::Tensor& coo_t);

torch::Tensor CooTensorToTorch(const Tensor& coo_t);

}  // namespace pytorch
}  // namespace dadt

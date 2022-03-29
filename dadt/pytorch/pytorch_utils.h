#pragma once

#include <torch/extension.h>
#include <torch/torch.h>

#include "t/element_type.h"
#include "t/shape.h"

namespace dadt {
namespace pytorch {

ElementType TorchDTypeToElementType(torch::Dtype dtype);

torch::Dtype ElementTypeToTorchDType(ElementType etype);

Shape TorchSizesToShape(const torch::IntArrayRef& sizes);

torch::IntArrayRef ShapeToTorchSizes(const Shape& shape);

}  // namespace pytorch
}  // namespace dadt

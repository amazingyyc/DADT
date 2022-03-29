#include "pytorch_utils.h"

#include "common/exception.h"

namespace dadt {
namespace pytorch {

ElementType TorchDTypeToElementType(torch::Dtype dtype) {
  switch (dtype) {
    case torch::kUInt8:
      return ElementType::From<uint8_t>();
    case torch::kInt8:
      return ElementType::From<int8_t>();
    case torch::kInt16:
      return ElementType::From<int16_t>();
    case torch::kInt32:
      return ElementType::From<int32_t>();
    case torch::kInt64:
      return ElementType::From<int64_t>();
    case torch::kFloat16:
      return ElementType::From<half>();
    case torch::kFloat32:
      return ElementType::From<float>();
    case torch::kFloat64:
      return ElementType::From<double>();
    default:
      return ElementType::From<UnKnown>();
  }
}

torch::Dtype ElementTypeToTorchDType(ElementType etype) {
  switch (etype.dtype) {
    case DType::kUint8:
      return torch::kUInt8;
    case DType::kInt8:
      return torch::kInt8;
    case DType::kInt16:
      return torch::kInt16;
    case DType::kInt32:
      return torch::kInt32;
    case DType::kInt64:
      return torch::kInt64;
    case DType::kFloat16:
      return torch::kFloat16;
    case DType::kFloat32:
      return torch::kFloat32;
    case DType::kFloat64:
      return torch::kFloat64;
    default:
      RUNTIME_ERROR("The ElementType does not support:" << etype.Name());
  }
}

Shape TorchSizesToShape(const torch::IntArrayRef& sizes) {
  std::vector<int64_t> dims = sizes.vec();
  return Shape(dims);
}

torch::IntArrayRef ShapeToTorchSizes(const Shape& shape) {
  const auto& dims = shape.dims();
  return torch::IntArrayRef(dims);
}

}  // namespace pytorch
}  // namespace dadt
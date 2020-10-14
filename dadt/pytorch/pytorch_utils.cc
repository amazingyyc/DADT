#include "pytorch_utils.h"

namespace dadt {
namespace pytorch {

dadt::ElementType get_element_type(const torch::Tensor &x) {
  return parse_element_type(x);
}

dadt::Shape get_shape_vector(const torch::Tensor &x) {
  return parse_shape_vector(x);
}

dadt::ElementType parse_element_type(const torch::Tensor &x) {
  switch (x.scalar_type()) {
    case torch::kByte:
      return dadt::ElementType::from<uint8_t>();
    case torch::kChar:
      return dadt::ElementType::from<int8_t>();
    case torch::kShort:
      return dadt::ElementType::from<int16_t>();
    case torch::kInt:
      return dadt::ElementType::from<int32_t>();
    case torch::kLong:
      return dadt::ElementType::from<int64_t>();
    case torch::kHalf:
      return dadt::ElementType::from<half>();
    case torch::kFloat:
      return dadt::ElementType::from<float>();
    case torch::kDouble:
      return dadt::ElementType::from<double>();
    default:
      RUNTIME_ERROR("the dtype does not support");
  }
}

dadt::Shape parse_shape_vector(const torch::Tensor &x) {
  std::vector<int> dims;

  for (auto d : x.sizes()) {
    ARGUMENT_CHECK(d > 0, "shape dim must > 0");

    dims.emplace_back(d);
  }

  return dadt::Shape(dims);
}

}
}
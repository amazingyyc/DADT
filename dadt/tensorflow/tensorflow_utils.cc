#include "tensorflow_utils.h"

// if the OpKernelContext is GPU
bool is_gpu_conext(OpKernelContext* context) {
  if (nullptr != context->device() &&
      nullptr != context->device()->tensorflow_gpu_device_info()) {
    return true;
  }

  return false;
}

dadt::ElementType convert_dtype_to_element_type(DataType dtype) {
  switch (dtype) {
    case DT_UINT8:
      return dadt::ElementType::from<uint8_t>();
    case DT_INT8:
      return dadt::ElementType::from<int8_t>();
    case DT_UINT16:
      return dadt::ElementType::from<uint16_t>();
    case DT_INT16:
      return dadt::ElementType::from<int16_t>();
    case DT_INT32:
      return dadt::ElementType::from<int32_t>();
    case DT_INT64:
      return dadt::ElementType::from<int64_t>();
    case DT_HALF:
      return dadt::ElementType::from<half>();
    case DT_FLOAT:
      return dadt::ElementType::from<float>();
    case DT_DOUBLE:
      return dadt::ElementType::from<double>();
    case DT_BOOL:
      return dadt::ElementType::from<bool>();
    default:
      RUNTIME_ERROR("the dtype does not support");
  }
}

std::vector<int> convert_tensor_shape_to_array(const TensorShape& shape) {
  std::vector<int> dims;

  for (auto d : shape) {
    ARGUMENT_CHECK(d.size > 0, "shape dim must > 0");

    dims.emplace_back(d.size);
  }

  return dims;
}
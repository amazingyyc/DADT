#include "pytorch_utils.h"

#include <memory>

#include "common/exception.h"
#include "pytorch_tensor_impl.h"

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
  std::vector<int64_t> dims(sizes.size());
  for (size_t i = 0; i < sizes.size(); ++i) {
    dims[i] = sizes[i];
  }

  return Shape(std::move(dims));
}

Tensor CooTensorFromTorch(const torch::Tensor& coo_t) {
  // Transpose from: [sparse_dim, nnz] to [nnz, sparse_dim].
  Tensor indices = Tensor(std::make_shared<PytorchTensorImpl>(
      coo_t.indices().transpose(0, 1).contiguous()));
  Tensor values =
      Tensor(std::make_shared<PytorchTensorImpl>(coo_t.values().contiguous()));

  Shape shape = TorchSizesToShape(coo_t.sizes());

  return Tensor::CooTensor(indices, values, shape, coo_t.is_coalesced());
}

Tensor CooTensorFromTorchClone(const torch::Tensor& coo_t) {
  Tensor indices = Tensor(std::make_shared<PytorchTensorImpl>(
      coo_t.indices().clone().transpose(0, 1).contiguous()));
  Tensor values = Tensor(
      std::make_shared<PytorchTensorImpl>(coo_t.values().clone().contiguous()));

  Shape shape = TorchSizesToShape(coo_t.sizes());

  return Tensor::CooTensor(indices, values, shape, coo_t.is_coalesced());
}

torch::Tensor CooTensorToTorch(const Tensor& coo_t) {
  auto indices = dynamic_cast<PytorchTensorImpl*>(coo_t.indices().impl().get())
                     ->torch_tensor();
  auto values = dynamic_cast<PytorchTensorImpl*>(coo_t.values().impl().get())
                    ->torch_tensor();

  Shape shape = coo_t.shape();

  // The IntArrayRef doesn't own the storage.
  torch::IntArrayRef sizes(shape.dims());

  return torch::sparse_coo_tensor(indices.transpose(0, 1), values, sizes)
      .coalesce();
}

}  // namespace pytorch
}  // namespace dadt
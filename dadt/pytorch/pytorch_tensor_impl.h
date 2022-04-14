#pragma once

#include <torch/extension.h>
#include <torch/torch.h>

#include <memory>

#include "common/exception.h"
#include "t/shape.h"
#include "t/tensor_impl.h"

namespace dadt {
namespace pytorch {

// ref:torch/include/ATen/core/TensorBody.h
// ref about cuda strema: https://github.com/pytorch/pytorch/issues/31696
class PytorchTensorImpl : public TensorImpl {
private:
  torch::Tensor torch_tensor_;

public:
  explicit PytorchTensorImpl(const torch::Tensor& torch_tensor);

  ~PytorchTensorImpl() = default;

public:
  const torch::Tensor& torch_tensor() const;

  bool IsDense() const override;

  bool IsCoo() const override;

  bool IsCpu() const override;

  bool IsCuda() const override;

  int DeviceId() const override;

  bool IsContiguous() const override;

  // The element type
  ElementType element_type() const override;

  // The tensor shape.
  Shape shape() const override;

  // Get tensor element count
  int64_t Size() const override;

  // Memory size
  size_t NumBytes() const override;

  // get memory pointer
  void* Ptr() override;
  void* Ptr() const override;

  std::shared_ptr<TensorImpl> DynamicZero(
      const Shape& shape, ElementType element_type) const override;
};

}  // namespace pytorch
}  // namespace dadt

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
class PytorchTensorImpl : public TensorImpl {
private:
  torch::Tensor torch_tensor_;

#ifdef HAVE_NCCL
  cudaEvent_t cuda_event_;
#endif

public:
  explicit PytorchTensorImpl(torch::Tensor torch_tensor);

  ~PytorchTensorImpl();

public:
  torch::Tensor torch_tensor() const;

#ifdef HAVE_NCCL
  cudaEvent_t cuda_event() const;
#endif

  bool IsCoo() const override;

  // Works for Coo tensor.
  std::shared_ptr<TensorImpl> Indices() const override;

  // Works for Coo tensor.
  std::shared_ptr<TensorImpl> Values() const override;

  int64_t SparseDim() const override;

  int64_t DenseDim() const override;

  int64_t nnz() const override;

  bool IsCoalesced() const override;

  bool IsDense() const override;

  // return GPU device id is CPU return -1
  int DeviceId() const override;

  // Whether the memory is continues
  bool IsContiguous() const override;

  // whether is a cpu tensor
  bool IsCpu() const override;

  // whether is a cuda tensor
  bool IsCuda() const override;

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

  // Dynamic mean the indices/values is: "PytorchTensorImpl" type.
  std::shared_ptr<TensorImpl> DynamicCoo(std::shared_ptr<TensorImpl> indices,
                                         std::shared_ptr<TensorImpl> values,
                                         const Shape& shape) const override;
};

}  // namespace pytorch
}  // namespace dadt

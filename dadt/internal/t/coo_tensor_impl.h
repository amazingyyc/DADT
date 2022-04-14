#pragma once

#include <cinttypes>

#include "t/element_type.h"
#include "t/shape.h"
#include "t/tensor.h"
#include "t/tensor_impl.h"

namespace dadt {

// The CooTensor is like pytorch's
// ref:https://pytorch.org/docs/stable/sparse.html But has same difference, the
// pytorch's Coo's indices shape is:[sparse_dim, nnz] But int this the indices's
// shape is:[nnz, sparse_dim], and the is_coalesced_ must be true.
class CooTensorImpl : public TensorImpl {
private:
  Tensor indices_;
  Tensor values_;

  Shape shape_;

  int64_t sparse_dim_;
  int64_t dense_dim_;
  int64_t nnz_;

  bool is_coalesced_;

public:
  CooTensorImpl(const Tensor& indices, const Tensor& values, const Shape& shape,
                bool is_coalesced);

public:
  // Whether is Dense tensor.
  bool IsDense() const override;

  // Whether is Coo tensor.
  bool IsCoo() const override;

  // whether is a cpu tensor, for Coo the indices and values must on same
  // device.
  bool IsCpu() const override;

  // whether is a cuda tensor, for Coo the indices and values must on same
  // device.
  bool IsCuda() const override;

  int DeviceId() const override;

  // Whether the memory is continues, only for Dense Tensor.
  bool IsContiguous() const override;

  // The element type, only for Dense Tensor.
  ElementType element_type() const override;

  // The tensor shape, only for Dense Tensor.
  Shape shape() const override;

  // Get tensor element count, only for Dense Tensor.
  int64_t Size() const override;

  // Memory bytes, only for Dense Tensor.
  size_t NumBytes() const override;

  // Memory pointer, only for Dense Tensor.
  void* Ptr() override;
  void* Ptr() const override;

  // The sparse dimension, only for Coo tensor.
  int64_t sparse_dim() const override;

  // The dense dimension, only for Coo tensor.
  int64_t dense_dim() const override;

  // The none zero count, only for Coo tensor.
  int64_t nnz() const override;

  // Whether coalesced, Only for Coo tensor.
  bool is_coalesced() const override;

  const Tensor& indices() const override;

  const Tensor& values() const override;

  std::shared_ptr<TensorImpl> DynamicZero(
      const Shape& shape, ElementType element_type) const override;
};

}  // namespace dadt

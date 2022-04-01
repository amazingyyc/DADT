#pragma once

#include <memory>

#include "t/element_type.h"
#include "t/shape.h"

namespace dadt {

class TensorImpl;

class Tensor {
private:
  std::shared_ptr<TensorImpl> impl_;

public:
  explicit Tensor(std::shared_ptr<TensorImpl> impl);

  ~Tensor() = default;

public:
  std::shared_ptr<TensorImpl> impl() const;

  bool IsDense() const;

  bool IsCoo() const;

  bool IsCpu() const;

  bool IsCuda() const;

  int DeviceId() const;

  bool IsContiguous() const;

  ElementType element_type() const;

  Shape shape() const;

  int64_t Size() const;

  size_t NumBytes() const;

  void* Ptr();
  void* Ptr() const;

  template <typename T>
  T* Data() {
    return (T*)Ptr();
  }

  template <typename T>
  T* Data() const {
    return (T*)Ptr();
  }

  int64_t sparse_dim() const;

  // The dense dimension, only for Coo tensor.
  int64_t dense_dim() const;

  int64_t nnz() const;

  bool is_coalesced() const;

  const Tensor& indices() const;

  const Tensor& values() const;

  // Dynamic means the returned Tensor is the same impl type with this.
  // Like If this Tensor contains a PytorchTensorImpl then it will returned
  // PytorchTensorImpl as impl_.
  Tensor DynamicZero(const Shape& shape, ElementType element_type) const;

public:
  static Tensor CooTensor(const Tensor& indices, const Tensor& values,
                          const Shape& shape, bool is_coalesced);
};

}  // namespace dadt

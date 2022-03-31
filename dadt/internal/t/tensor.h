#pragma once

#include <memory>

#include "t/element_type.h"
#include "t/shape.h"
#include "t/tensor_impl.h"

namespace dadt {

class Tensor {
private:
  std::shared_ptr<TensorImpl> impl_;

public:
  explicit Tensor(std::shared_ptr<TensorImpl> impl);

  virtual ~Tensor() = default;

public:
  std::shared_ptr<TensorImpl> impl() const;

  bool IsCoo() const;

  // Works for Coo tensor.
  Tensor Indices() const;

  // Works for Coo tensor.
  Tensor Values() const;

  int64_t SparseDim() const;

  int64_t DenseDim() const;

  // Non zero count of Coo.
  int64_t nnz() const;

  bool IsCoalesced() const;

  bool IsDense() const;

  // return GPU device id is CPU return -1
  int DeviceId() const;

  // Whether the memory is continues
  bool IsContiguous() const;

  // whether is a cpu tensor
  bool IsCpu() const;

  // whether is a cuda tensor
  bool IsCuda() const;

  // The element type
  ElementType element_type() const;

  Shape shape() const;

  // Get tensor element count
  int64_t Size() const;

  // Memory size
  size_t NumBytes() const;

  // get memory pointer
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

  Tensor Transpose(int64_t dim0, int64_t dim1) const;

  Tensor Coalesce() const;

#ifdef HAVE_NCCL
  std::unique_ptr<StreamGuard> DynamicCudaStreamGuard(
      cudaStream_t cuda_stream, int8_t device_index) const;
#endif

  // Dynamic means the returned Tensor is the same impl type with this.
  // Like If this Tensor contains a PytorchTensorImpl then it will returned a
  // PytorchTensorImpl as impl_.
  Tensor DynamicZero(const Shape& shape, ElementType element_type) const;

  // The indices/values/this must has same implement type.
  Tensor DynamicCoo(const Tensor& indices, const Tensor& values,
                    const Shape& shape) const;
};

}  // namespace dadt

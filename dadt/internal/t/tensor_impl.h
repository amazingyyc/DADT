#pragma once

#include <cinttypes>

#include "t/element_type.h"
#include "t/shape.h"

namespace dadt {

class TensorImpl {
public:
  virtual ~TensorImpl() = default;

  virtual bool IsCoo() const = 0;

  // Works for Coo tensor.
  virtual std::shared_ptr<TensorImpl> Indices() const = 0;

  // Works for Coo tensor.
  virtual std::shared_ptr<TensorImpl> Values() const = 0;

  // Works for Coo tensor.
  virtual int64_t SparseDim() const = 0;

  // Works for Coo tensor.
  virtual int64_t DenseDim() const = 0;

  // Works for Coo tensor.
  virtual int64_t nnz() const = 0;

  // Works for Coo tensor.
  virtual bool IsCoalesced() const = 0;

  virtual bool IsDense() const = 0;

  // return GPU device id is CPU return -1
  virtual int DeviceId() const = 0;

  // Whether the memory is continues
  virtual bool IsContiguous() const = 0;

  // whether is a cpu tensor
  virtual bool IsCpu() const = 0;

  // whether is a cuda tensor
  virtual bool IsCuda() const = 0;

  // The element type
  virtual ElementType element_type() const = 0;

  virtual Shape shape() const = 0;

  // Get tensor element count
  virtual int64_t Size() const = 0;

  // Memory size
  virtual size_t NumBytes() const = 0;

  // get memory pointer
  virtual void* Ptr() = 0;
  virtual void* Ptr() const = 0;

  virtual std::shared_ptr<TensorImpl> DynamicZero(
      const Shape& shape, ElementType element_type) const = 0;

  virtual std::shared_ptr<TensorImpl> DynamicCoo(
      std::shared_ptr<TensorImpl> indices, std::shared_ptr<TensorImpl> values,
      const Shape& shape) const = 0;
};

}  // namespace dadt

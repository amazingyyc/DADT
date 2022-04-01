#pragma once

#include <cinttypes>

#include "common/exception.h"
#include "common/stream_guard.h"
#include "t/element_type.h"
#include "t/shape.h"
#include "t/tensor.h"

namespace dadt {

class TensorImpl {
public:
  virtual ~TensorImpl() = default;

  // Whether is Dense tensor.
  virtual bool IsDense() const = 0;

  // Whether is Coo tensor.
  virtual bool IsCoo() const = 0;

  // whether is a cpu tensor, for Coo the indices and values must on same
  // device.
  virtual bool IsCpu() const = 0;

  // whether is a cuda tensor, for Coo the indices and values must on same
  // device.
  virtual bool IsCuda() const = 0;

  // The device id: only for Dense Tensor.
  virtual int DeviceId() const = 0;

  // Whether the memory is continues, only for Dense Tensor.
  virtual bool IsContiguous() const = 0;

  // The element type, only for Dense Tensor.
  virtual ElementType element_type() const = 0;

  // The tensor shape, only for Dense Tensor.
  virtual Shape shape() const = 0;

  // Get tensor element count, only for Dense Tensor.
  virtual int64_t Size() const = 0;

  // Memory bytes, only for Dense Tensor.
  virtual size_t NumBytes() const = 0;

  // Memory pointer, only for Dense Tensor.
  virtual void* Ptr() = 0;
  virtual void* Ptr() const = 0;

  // The sparse dimension, only for Coo tensor.
  virtual int64_t sparse_dim() const {
    RUNTIME_ERROR("UnSupport funciton");
  }

  // The dense dimension, only for Coo tensor.
  virtual int64_t dense_dim() const {
    RUNTIME_ERROR("UnSupport funciton");
  }

  // The none zero count, only for Coo tensor.
  virtual int64_t nnz() const {
    RUNTIME_ERROR("UnSupport funciton");
  }

  // Whether coalesced, Only for Coo tensor.
  virtual bool is_coalesced() const {
    RUNTIME_ERROR("UnSupport funciton");
  }

  virtual const Tensor& indices() const {
    RUNTIME_ERROR("UnSupport funciton");
  }

  virtual const Tensor& values() const {
    RUNTIME_ERROR("UnSupport funciton");
  }

  virtual std::shared_ptr<TensorImpl> DynamicZero(
      const Shape& shape, ElementType element_type) const = 0;
};

}  // namespace dadt

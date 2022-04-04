#include "t/coo_tensor_impl.h"

#include "common/exception.h"
#include "t/tensor_impl.h"

namespace dadt {

CooTensorImpl::CooTensorImpl(const Tensor& indices, const Tensor& values,
                             const Shape& shape, bool is_coalesced)
    : indices_(indices),
      values_(values),
      shape_(shape),
      is_coalesced_(is_coalesced) {
  ARGUMENT_CHECK(indices_.DeviceId() == values_.DeviceId(),
                 "CooTensor need indices/values on same device");
  ARGUMENT_CHECK(indices_.IsDense() && values_.IsDense(),
                 "CooTensor need indices/values is DenseTensor");
  ARGUMENT_CHECK(indices_.element_type().Is<int64_t>(),
                 "CooTensor need indices ElementType is int64");

  Shape indices_shape = indices_.shape();
  Shape values_shape = values.shape();

  ARGUMENT_CHECK(indices_shape.NDims() == 2,
                 "CooTensor need indices's NDim is 2.");
  ARGUMENT_CHECK(
      indices_shape[0] == values_shape[0],
      "CooTensor need indices's first dimension same with values's first "
      "dimension.");

  ARGUMENT_CHECK(shape_.NDims() + 1 == indices_shape[1] + values_shape.NDims(),
                 "CooTensor shape error!");

  nnz_ = indices_shape[0];
  sparse_dim_ = indices_shape[1];

  for (int64_t i = sparse_dim_, j = 1; i < shape_.NDims(); ++i, ++j) {
    ARGUMENT_CHECK(shape_[i] == values_shape[j], "CooTensor shape error!");
  }

  dense_dim_ = values_shape.NDims() - 1;
}

bool CooTensorImpl::IsDense() const {
  return false;
}

bool CooTensorImpl::IsCoo() const {
  return true;
}

bool CooTensorImpl::IsCpu() const {
  ARGUMENT_CHECK(indices_.IsCpu() == values_.IsCpu(),
                 "CooTensor need indices/valeus on same device");

  return indices_.IsCpu();
}

bool CooTensorImpl::IsCuda() const {
  ARGUMENT_CHECK(indices_.IsCuda() == values_.IsCuda(),
                 "CooTensor need indices/valeus on same device");

  return indices_.IsCuda();
}

int CooTensorImpl::DeviceId() const {
  ARGUMENT_CHECK(indices_.DeviceId() == values_.DeviceId(),
                 "CooTensor need indices/valeus on same device");

  return indices_.DeviceId();
}

bool CooTensorImpl::IsContiguous() const {
  return indices_.IsContiguous() && indices_.IsContiguous();
}

ElementType CooTensorImpl::element_type() const {
  RUNTIME_ERROR("CooTensor not support function: element_type");
}

Shape CooTensorImpl::shape() const {
  return shape_;
}

int64_t CooTensorImpl::Size() const {
  RUNTIME_ERROR("CooTensor not support function: shape");
}

size_t CooTensorImpl::NumBytes() const {
  RUNTIME_ERROR("CooTensor not support function: NumBytes");
}

void* CooTensorImpl::Ptr() {
  RUNTIME_ERROR("CooTensor not support function: Ptr");
}

void* CooTensorImpl::Ptr() const {
  RUNTIME_ERROR("CooTensor not support function: Ptr");
}

int64_t CooTensorImpl::sparse_dim() const {
  return sparse_dim_;
}

int64_t CooTensorImpl::dense_dim() const {
  return dense_dim_;
}

int64_t CooTensorImpl::nnz() const {
  return nnz_;
}

bool CooTensorImpl::is_coalesced() const {
  return is_coalesced_;
}

const Tensor& CooTensorImpl::indices() const {
  return indices_;
}

const Tensor& CooTensorImpl::values() const {
  return values_;
}

std::shared_ptr<TensorImpl> CooTensorImpl::DynamicZero(
    const Shape& shape, ElementType element_type) const {
  RUNTIME_ERROR("CooTensor not support function: DynamicZero");
}

}  // namespace dadt

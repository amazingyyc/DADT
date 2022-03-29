#include "t/tensor.h"

namespace dadt {

Tensor::Tensor(std::shared_ptr<TensorImpl> impl) : impl_(impl) {
}

std::shared_ptr<TensorImpl> Tensor::impl() const {
  return impl_;
}

bool Tensor::IsCoo() const {
  return impl_->IsCoo();
}

Tensor Tensor::Indices() const {
  return Tensor(impl_->Indices());
}

Tensor Tensor::Values() const {
  return Tensor(impl_->Values());
}

int64_t Tensor::SparseDim() const {
  return impl_->SparseDim();
}

int64_t Tensor::DenseDim() const {
  return impl_->DenseDim();
}

int64_t Tensor::nnz() const {
  return impl_->nnz();
}

bool Tensor::IsCoalesced() const {
  return impl_->IsCoalesced();
}

bool Tensor::IsDense() const {
  return impl_->IsDense();
}

int Tensor::DeviceId() const {
  return impl_->DeviceId();
}

bool Tensor::IsContiguous() const {
  return impl_->IsContiguous();
}

bool Tensor::IsCpu() const {
  return impl_->IsCpu();
}

bool Tensor::IsCuda() const {
  return impl_->IsCuda();
}

ElementType Tensor::element_type() const {
  return impl_->element_type();
}

Shape Tensor::shape() const {
  return impl_->shape();
}

int64_t Tensor::Size() const {
  return impl_->Size();
}

size_t Tensor::NumBytes() const {
  return impl_->NumBytes();
}

void* Tensor::Ptr() {
  return impl_->Ptr();
}

void* Tensor::Ptr() const {
  return impl_->Ptr();
}

Tensor Tensor::DynamicZero(const Shape& shape, ElementType element_type) const {
  return Tensor(impl_->DynamicZero(shape, element_type));
}

Tensor Tensor::DynamicCoo(const Tensor& indices, const Tensor& values,
                          const Shape& shape) const {
  auto cool_impl = impl_->DynamicCoo(indices.impl_, values.impl_, shape);

  return Tensor(cool_impl);
}

}  // namespace dadt

#include "t/tensor.h"

#include "t/coo_tensor_impl.h"
#include "t/tensor_impl.h"

namespace dadt {

Tensor::Tensor(std::shared_ptr<TensorImpl> impl) : impl_(impl) {
}

std::shared_ptr<TensorImpl> Tensor::impl() const {
  return impl_;
}

bool Tensor::IsDense() const {
  return impl_->IsDense();
}

bool Tensor::IsCoo() const {
  return impl_->IsCoo();
}

bool Tensor::IsCpu() const {
  return impl_->IsCpu();
}

bool Tensor::IsCuda() const {
  return impl_->IsCuda();
}

int Tensor::DeviceId() const {
  return impl_->DeviceId();
}

bool Tensor::IsContiguous() const {
  return impl_->IsContiguous();
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

int64_t Tensor::sparse_dim() const {
  return impl_->sparse_dim();
}

int64_t Tensor::dense_dim() const {
  return impl_->dense_dim();
}

int64_t Tensor::nnz() const {
  return impl_->nnz();
}

bool Tensor::is_coalesced() const {
  return impl_->is_coalesced();
}

const Tensor& Tensor::indices() const {
  return impl_->indices();
}

const Tensor& Tensor::values() const {
  return impl_->values();
}

Tensor Tensor::DynamicZero(const Shape& shape, ElementType element_type) const {
  return Tensor(impl_->DynamicZero(shape, element_type));
}

Tensor Tensor::CooTensor(const Tensor& indices, const Tensor& values,
                         const Shape& shape, bool is_coalesced) {
  auto impl =
      std::make_shared<CooTensorImpl>(indices, values, shape, is_coalesced);

  return Tensor(impl);
}

}  // namespace dadt

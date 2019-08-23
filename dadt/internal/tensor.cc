#include "tensor.h"

namespace dadt {

Tensor::Tensor(std::shared_ptr<TensorStorage> storage, size_t offset, Shape shape, ElementType element_type)
:storage_(storage), offset_(offset), shape_(shape), element_type_(element_type) {
}

std::shared_ptr<Device> Tensor::device() {
  return storage_->device();
}

size_t Tensor::offset() const {
  return offset_;
}

const Shape &Tensor::shape() const {
  return shape_;
}

const ElementType &Tensor::element_type() const {
  return element_type_;
}

void *Tensor::ptr() {
  return ((uint8_t *)storage_->ptr()) + offset_;
}

void *Tensor::ptr() const {
  return ((uint8_t *)storage_->ptr()) + offset_;
}

bool Tensor::is_scalar() const {
  return shape_.is_scalar();
}

int Tensor::size() const {
  return shape_.size();
}

int Tensor::num_bytes() const {
  return element_type_.byte_width() * size();
}

int Tensor::dim(int axis) const {
  return shape_.dim(axis);
}

void Tensor::copy_from_cpu(const void *data) {
  this->device()->memcpy_from_cpu(this->ptr(), data, this->num_bytes());
}

void Tensor::copy_from_gpu(const void *data) {
  this->device()->memcpy_from_gpu(this->ptr(), data, this->num_bytes());
}

void Tensor::copy_to_cpu(void *data) {
  this->device()->memcpy_to_cpu(data, this->ptr(), this->num_bytes());
}

void Tensor::copy_to_gpu(void *data) {
  this->device()->memcpy_to_gpu(data, this->ptr(), this->num_bytes());
}

}
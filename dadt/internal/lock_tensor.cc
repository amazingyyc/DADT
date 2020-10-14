#include <iostream>
#include <string>

#include "definition.h"
#include "lock_tensor.h"

namespace dadt {

LockTensor::LockTensor(
  std::shared_ptr<TensorStorage> storage,
  size_t offset,
  Shape shape,
  ElementType element_type,
  std::string name,
  LockTensorStatus initialize_status)
  :storage_(storage),
  offset_(offset),
  shape_(shape),
  element_type_(element_type),
  name_(name),
  status_((int)initialize_status) {
}

void LockTensor::wait(LockTensorStatus expected_status, LockTensorStatus new_status) {
  status_.wait((int)expected_status, (int)new_status);
}

const std::string& LockTensor::name() const {
  return name_;
}

const Shape& LockTensor::shape() const {
  return shape_;
}

const ElementType& LockTensor::element_type() const {
  return element_type_;
}

bool LockTensor::is_scalar() const {
  return shape_.is_scalar();
}

int LockTensor::size() const {
  return shape_.size();
}

int LockTensor::num_bytes() const {
  return element_type_.byte_width() * size();
}

int LockTensor::dim(int axis) const {
  return shape_.dim(axis);
}

bool LockTensor::is_cpu() const {
  return !is_cuda();
}

int LockTensor::device_id() const {
  if (is_cpu()) {
    return -1;
  }

  return storage_->device()->device_id();
}

// whether is a cuda tensor
bool LockTensor::is_cuda() const {
  return DeviceType::GPU == storage_->device()->device_type();
}

void* LockTensor::ptr() {
  ((uint8_t *)storage_->ptr()) + offset_;
}

void* LockTensor::ptr() const {
  ((uint8_t *)storage_->ptr()) + offset_;
}

}
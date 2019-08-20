#include "tensor_storage.h"

namespace dadt {

TensorStorage::TensorStorage(std::shared_ptr<Device> device, void *ptr, size_t size)
  : device_(device), ptr_(ptr), size_(size) {
}

TensorStorage::~TensorStorage() {
  device_->free(ptr_);

  ptr_ = nullptr;
  size_ = 0;
}

std::shared_ptr<Device> TensorStorage::device() {
  return device_;
}

void *TensorStorage::ptr() {
  return ptr_;
}

size_t TensorStorage::size() {
  return size_;
}

// create a tensorstorage from a special device
std::shared_ptr<TensorStorage> TensorStorage::create(std::shared_ptr<Device> device, size_t size) {
  void *ptr = device->malloc(size);

  // set to be zero
  device->zero(ptr, size);

  return std::make_shared<TensorStorage>(device, ptr, size);
}

}
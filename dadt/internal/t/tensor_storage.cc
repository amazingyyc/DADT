#include "t/tensor_storage.h"

namespace dadt {

TensorStorage::TensorStorage(Device* device, void* ptr, size_t size)
    : device_(device), ptr_(ptr), size_(size) {
}

TensorStorage::~TensorStorage() {
  device_->Free(ptr_);

  ptr_ = nullptr;
  size_ = 0;
}

Device* TensorStorage::device() {
  return device_;
}

void* TensorStorage::ptr() {
  return ptr_;
}

size_t TensorStorage::size() {
  return size_;
}

// create a tensorstorage from a special device
std::shared_ptr<TensorStorage> TensorStorage::Create(Device* device,
                                                     size_t size) {
  void* ptr = device->Malloc(size);
  device->Zero(ptr, size);

  return std::make_shared<TensorStorage>(device, ptr, size);
}

}  // namespace dadt
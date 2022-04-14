#include "common/buffer.h"

namespace dadt {

Buffer::Buffer(Device* device) : device_(device), ptr_(nullptr), size_(0) {
}

Buffer::~Buffer() {
  if (nullptr != ptr_) {
    device_->Free(ptr_);
  }

  ptr_ = nullptr;
  size_ = 0;
}

void* Buffer::ptr(size_t offset) {
  return ((uint8_t*)ptr_) + offset;
}

size_t Buffer::size() {
  return size_;
}

void Buffer::Reserve(size_t new_size) {
  if (new_size > size_) {
    if (nullptr != ptr_) {
      device_->Free(ptr_);
    }

    ptr_ = device_->Malloc(new_size);
    size_ = new_size;
  }
}

void Buffer::Zero() {
  if (nullptr != ptr_) {
    device_->Zero(ptr_, size_);
  }
}

}  // namespace dadt

#include "definition.h"
#include "memory_buffer.h"

namespace dadt {

MemoryBuffer::MemoryBuffer(std::shared_ptr<Device> device): device_(device), ptr_(nullptr), size_(0) {
}

void* MemoryBuffer::ptr(size_t offset) {
  return ((uint8_t*)ptr_) + offset;
}

size_t MemoryBuffer::size() {
  return size_;
}

MemoryBuffer::~MemoryBuffer() {
  if (nullptr != ptr_) {
    device_->free(ptr_);
  }

  ptr_  = nullptr;
  size_ = 0;
}

void MemoryBuffer::reserve(size_t new_size) {
  if (new_size > size_) {
    if (nullptr != ptr_) {
      device_->free(ptr_);
    }

    ptr_  = nullptr;
    size_ = 0;

    ptr_  = device_->malloc(new_size);
    size_ = new_size;
  }
}
}
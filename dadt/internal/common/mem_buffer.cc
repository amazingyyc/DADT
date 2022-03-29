#include "common/mem_buffer.h"

#include <cinttypes>
#include <cstdlib>

namespace dadt {

MemBuffer::MemBuffer(size_t capacity) : capacity_(capacity) {
  ptr_ = std::malloc(capacity_);
}

MemBuffer::~MemBuffer() {
  std::free(ptr_);
}

void* MemBuffer::ptr(size_t offset) const {
  return (uint8_t*)(ptr_) + offset;
}

size_t MemBuffer::capacity() const {
  return capacity_;
}

}  // namespace dadt

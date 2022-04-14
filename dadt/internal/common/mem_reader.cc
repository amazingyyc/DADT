#include "common/mem_reader.h"

#include <cstring>

namespace dadt {

MemReader::MemReader(const char* ptr, size_t length)
    : ptr_(ptr), length_(length), offset_(0) {
}

bool MemReader::Read(void* target, size_t size) {
  if (ptr_ == nullptr || offset_ + size > length_) {
    return false;
  }

  memcpy(target, ptr_ + offset_, size);
  offset_ += size;

  return true;
}

}  // namespace dadt

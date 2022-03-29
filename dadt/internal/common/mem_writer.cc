#include "common/mem_writer.h"

#include <cstdlib>
#include <cstring>

namespace dadt {

MemWriter::MemWriter() : ptr_(nullptr), capacity_(0), offset_(0) {
}

MemWriter::MemWriter(size_t expect) : ptr_(nullptr), offset_(0) {
  ptr_ = (char*)std::malloc(expect);
  capacity_ = expect;
}

MemWriter::MemWriter(MemWriter&& other)
    : ptr_(other.ptr_), capacity_(other.capacity_), offset_(other.offset_) {
  other.ptr_ = nullptr;
  other.capacity_ = 0;
  other.offset_ = 0;
}

const MemWriter& MemWriter::operator=(MemWriter&& other) {
  if (ptr_ != nullptr) {
    std::free(ptr_);

    ptr_ = nullptr;
    capacity_ = 0;
    offset_ = 0;
  }

  ptr_ = other.ptr_;
  capacity_ = other.capacity_;
  offset_ = other.offset_;

  other.ptr_ = nullptr;
  other.capacity_ = 0;
  other.offset_ = 0;

  return *this;
}

MemWriter::~MemWriter() {
  if (ptr_ != nullptr) {
    std::free(ptr_);
  }

  ptr_ = nullptr;
  capacity_ = 0;
  offset_ = 0;
}

void MemWriter::Growth(size_t new_size) {
  size_t new_capacity = capacity_ + capacity_ / 2;

  if (new_capacity < new_size) {
    new_capacity = new_size;
  }

  char* new_ptr = (char*)std::malloc(new_capacity);

  // copy old data
  if (offset_ > 0) {
    std::memcpy(new_ptr, ptr_, offset_);
  }

  if (ptr_ != nullptr) {
    std::free(ptr_);
  }

  ptr_ = new_ptr;
  capacity_ = new_capacity;
}

char* MemWriter::ptr() const {
  return ptr_;
}

size_t MemWriter::capacity() const {
  return capacity_;
}

size_t MemWriter::offset() const {
  return offset_;
}

bool MemWriter::Write(const char* bytes, size_t size) {
  if (ptr_ == nullptr || offset_ + size > capacity_) {
    Growth(offset_ + size);
  }

  std::memcpy(ptr_ + offset_, bytes, size);
  offset_ += size;

  return true;
}

}  // namespace dadt

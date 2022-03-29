#pragma once

#include "common/ireader.h"

namespace dadt {

class MemBuffer {
private:
  void* ptr_;
  size_t capacity_;

public:
  MemBuffer(size_t capacity);

  ~MemBuffer();

  void* ptr(size_t offset = 0) const;

  size_t capacity() const;
};

}  // namespace dadt

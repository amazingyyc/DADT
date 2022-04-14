#pragma once

#include <cstdlib>

#include "common/ireader.h"

namespace dadt {

class MemReader : public IReader {
private:
  const char* ptr_;
  size_t length_;
  size_t offset_;

public:
  MemReader(const char* ptr, size_t length);

  bool Read(void* target, size_t size) override;
};

}  // namespace dadt

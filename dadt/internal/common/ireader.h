#pragma once

#include <cstddef>

namespace dadt {

class IReader {
public:
  virtual bool Read(void* target, size_t size) = 0;
};

}  // namespace dadt

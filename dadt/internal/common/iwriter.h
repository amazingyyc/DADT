#pragma once

#include <cstddef>

namespace dadt {

class IWriter {
public:
  virtual bool Write(const char* ptr, size_t size) = 0;
};

}  // namespace dadt

#pragma once

#include "common/device.h"

namespace dadt {

class Buffer {
private:
  // the device
  Device* device_;

  // memory pointer
  void* ptr_;

  // the memory size
  size_t size_;

public:
  Buffer(Device* device);

  ~Buffer();

  void* ptr(size_t offset = 0);

  size_t size();

  // the function is not threead-safe
  // reserve the size memory, if current size is small remalloc a new memory
  // or reuse it
  void Reserve(size_t);

  void Zero();
};

}  // namespace dadt

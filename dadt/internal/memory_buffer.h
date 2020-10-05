#ifndef MEMORY_BUFFER_H
#define MEMORY_BUFFER_H

#include <iostream>
#include <memory>

#include "device.h"
#include "lock_tensor.h"

namespace dadt {

class MemoryBuffer {
private:
  // the device
  std::shared_ptr<Device> device_;

  // memory pointer
  void *ptr_;

  // the memory size
  size_t size_;

public:
  MemoryBuffer(std::shared_ptr<Device> device);

  ~MemoryBuffer();

  void* ptr(size_t offset = 0);

  size_t size();

  // the function is not threead-safe
  // reserve the size memory, if current size is small remalloc a new memory
  // or reuse it
  void reserve(size_t);

  void zero();
};

}

#endif
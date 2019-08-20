#ifndef TENSOR_STORAGE_H
#define TENSOR_STORAGE_H

#include <iostream>

#include "device.h"

namespace dadt {

class TensorStorage: public std::enable_shared_from_this<TensorStorage> {
private:
    // the device that hold the memory
  std::shared_ptr<Device> device_;

  // memory pointer
  void *ptr_;

  // the memory size
  size_t size_;

public:
  TensorStorage(std::shared_ptr<Device>, void *, size_t);

  ~TensorStorage();

  std::shared_ptr<Device> device();

  void *ptr();

  size_t size();

public:
  // create a tensorstorage from a special device
  static std::shared_ptr<TensorStorage> create(std::shared_ptr<Device> device, size_t size);
};

}

#endif
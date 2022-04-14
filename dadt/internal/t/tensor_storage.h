#pragma once

#include <iostream>

#include "common/device.h"

namespace dadt {

class TensorStorage {
private:
  // the device that hold the memory
  Device* device_;

  // memory pointer
  void* ptr_;

  // the memory size
  size_t size_;

public:
  TensorStorage(Device*, void*, size_t);

  TensorStorage(const TensorStorage&) = delete;
  TensorStorage& operator=(const TensorStorage&) = delete;
  TensorStorage(TensorStorage&&) = delete;
  TensorStorage& operator=(TensorStorage&&) = delete;

  ~TensorStorage();

  Device* device();

  void* ptr();

  size_t size();

public:
  static std::shared_ptr<TensorStorage> Create(Device* device, size_t size);
};

}  // namespace dadt

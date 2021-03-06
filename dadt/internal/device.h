#ifndef DEVICE_H
#define DEVICE_H

#include <iostream>
#include <memory>
#include <cstring>

namespace dadt{

 // a device type, for now only CPU
enum class DeviceType: int {
  CPU = 0,
  GPU = 1,
};

// used to allocate memory from device
class IAllocator {
public:
  // malloc memory from device
  virtual void* malloc(size_t) = 0;

  // free memory to device
  virtual void free(void*) = 0;

  // zero the memory
  virtual void zero(void*, size_t) = 0;
};

class CPUAllocator: public IAllocator {
public:
  void *malloc(size_t) override;

  void free(void *) override;

  // zero the memory
  void zero(void*, size_t) override;
};

class GPUAllocator: public IAllocator {
private:
  // the gpu device id
  int device_id_;

public:
  GPUAllocator(int device_id);

  void *malloc(size_t) override;

  void free(void *) override;

  // zero the memory
  void zero(void*, size_t) override;
};

class Device: public std::enable_shared_from_this<Device> {
private:
  // the device id
  // CPU is -1
  // GPU is corresponding the real GPU device
  int device_id_;

  // device type
  DeviceType device_type_;

  // memory allocator
  std::shared_ptr<IAllocator> allocator_;

public:
  explicit Device(int, DeviceType);

  ~Device() = default;

  bool operator==(const Device &) const;

  int device_id();

  DeviceType device_type();

  // malloc memory from device
  void *malloc(size_t);

  // free the memory
  void free(void *);

  // zero the memory
  void zero(void*, size_t);
};

// this two funcion is thread-safe
std::shared_ptr<Device> get_cpu_device();
std::shared_ptr<Device> get_gpu_device(int device_id);

}

#endif
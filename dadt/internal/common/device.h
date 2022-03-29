#pragma once

#include <cstring>
#include <memory>

namespace dadt {

// a device type, for now only CPU
enum class DeviceType : uint8_t {
  kCPU = 0,
  kGPU = 1,
};

// used to allocate memory from device
class IAllocator {
public:
  // malloc memory from device
  virtual void* Malloc(size_t) = 0;

  // free memory to device
  virtual void Free(void*) = 0;

  // zero the memory
  virtual void Zero(void*, size_t) = 0;
};

class CPUAllocator : public IAllocator {
public:
  void* Malloc(size_t) override;

  void Free(void*) override;

  // zero the memory
  void Zero(void*, size_t) override;
};

class GPUAllocator : public IAllocator {
private:
  // the gpu device id
  int device_id_;

public:
  GPUAllocator(int device_id);

  void* Malloc(size_t) override;

  void Free(void*) override;

  // zero the memory
  void Zero(void*, size_t) override;
};

class Device {
private:
  // the device id
  // CPU is -1
  // GPU is corresponding the real GPU device
  int device_id_;

  // device type
  DeviceType device_type_;

  // memory allocator
  std::unique_ptr<IAllocator> allocator_;

public:
  Device(int, DeviceType);

  ~Device() = default;

  bool operator==(const Device&) const;

  int device_id();

  DeviceType device_type();

  // malloc memory from device
  void* Malloc(size_t);

  // free the memory
  void Free(void*);

  // zero the memory
  void Zero(void*, size_t);

public:
  static Device* DDevice(int device_id);
  static Device* CPUDevice();
  static Device* GPUDevice(int device_id);
};

// this two funcion is thread-safe
std::shared_ptr<Device> get_cpu_device();
std::shared_ptr<Device> get_gpu_device(int device_id);

}  // namespace dadt

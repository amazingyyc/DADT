#include "common/device.h"

#ifdef HAVE_NCCL
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#endif

#include <unordered_map>
#include <utility>

#include "common/exception.h"
#include "common/spin_lock.h"

namespace dadt {

void* CPUAllocator::Malloc(size_t size) {
  return std::malloc(size);
}

void CPUAllocator::Free(void* ptr) {
  std::free(ptr);
}

// zero the memory
void CPUAllocator::Zero(void* ptr, size_t size) {
  std::memset(ptr, 0, size);
}

GPUAllocator::GPUAllocator(int device_id) : device_id_(device_id) {
}

void* GPUAllocator::Malloc(size_t size) {
#ifdef HAVE_NCCL
  void* ptr;

  CUDA_CALL(cudaSetDevice(device_id_));
  CUDA_CALL(cudaMalloc(&ptr, size));

  return ptr;
#else
  RUNTIME_ERROR("Has no CUDA");
#endif
}

void GPUAllocator::Free(void* ptr) {
#ifdef HAVE_NCCL
  CUDA_CALL(cudaSetDevice(device_id_));
  CUDA_CALL(cudaFree(ptr));
#else
  RUNTIME_ERROR("Has no CUDA");
#endif
}

// zero the memory
void GPUAllocator::Zero(void* ptr, size_t size) {
#ifdef HAVE_NCCL
  CUDA_CALL(cudaSetDevice(device_id_));
  CUDA_CALL(cudaMemset(ptr, 0, size));
#else
  RUNTIME_ERROR("Has no CUDA");
#endif
}

Device::Device(int device_id, DeviceType device_type)
    : device_id_(device_id), device_type_(device_type) {
  if (DeviceType::kCPU == device_type_) {
    // CPU device always is -1
    device_id_ = -1;
    allocator_.reset(new CPUAllocator());
  } else if (DeviceType::kGPU == device_type_) {
    ARGUMENT_CHECK(device_id_ >= 0, "For GPU device, device_id must >= 0.")
    allocator_.reset(new GPUAllocator(device_id_));
  } else {
    RUNTIME_ERROR("Device type" << (int32_t)device_type << " is not support!");
  }
}

bool Device::operator==(const Device& other) const {
  return device_type_ == other.device_type_ && device_id_ == other.device_id_;
}

DeviceType Device::device_type() {
  return device_type_;
}

int Device::device_id() {
  return device_id_;
}

// malloc memory from device
void* Device::Malloc(size_t size) {
  return allocator_->Malloc(size);
}

// free the memory
void Device::Free(void* ptr) {
  allocator_->Free(ptr);
}

// zero the memory
void Device::Zero(void* ptr, size_t size) {
  allocator_->Zero(ptr, size);
}

Device* Device::DDevice(int device_id) {
  static SpinLock locker;
  static std::unordered_map<int, std::unique_ptr<Device>> devices;

  SpinLockHandler _(locker);

  DeviceType device_type;

  if (device_id < 0) {
    device_id = -1;
    device_type = DeviceType::kCPU;
  } else {
    device_type = DeviceType::kGPU;
  }

  if (devices.find(device_id) == devices.end()) {
    devices.emplace(
        device_id, std::unique_ptr<Device>(new Device(device_id, device_type)));
  }

  return devices[device_id].get();
}

Device* Device::CPUDevice() {
  return DDevice(-1);
}

Device* Device::GPUDevice(int device_id) {
  return DDevice(device_id);
}

}  // namespace dadt

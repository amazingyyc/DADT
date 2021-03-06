
#ifdef HAVE_NCCL
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#endif

#include <iostream>
#include <unordered_map>

#include "definition.h"
#include "device.h"
#include "spin_lock.h"

namespace dadt {

void* CPUAllocator::malloc(size_t size) {
  return std::malloc(size);
}

void CPUAllocator::free(void* ptr) {
  std::free(ptr);
}

// zero the memory
void CPUAllocator::zero(void *ptr, size_t size) {
  std::memset(ptr, 0, size);
}

GPUAllocator::GPUAllocator(int device_id): device_id_(device_id) {
}

void* GPUAllocator::malloc(size_t size) {
#ifdef HAVE_NCCL
  void *ptr;

  CUDA_CALL(cudaSetDevice(device_id_));
  CUDA_CALL(cudaMalloc(&ptr, size));

  return ptr;
#else
  RUNTIME_ERROR("do not have CUDA");
#endif
}

void GPUAllocator::free(void *ptr) {
#ifdef HAVE_NCCL
  CUDA_CALL(cudaSetDevice(device_id_));
  CUDA_CALL(cudaFree(ptr));
#else
  RUNTIME_ERROR("do not have CUDA");
#endif
}

// zero the memory
void GPUAllocator::zero(void *ptr, size_t size) {
  #ifdef HAVE_NCCL
  CUDA_CALL(cudaSetDevice(device_id_));
  CUDA_CALL(cudaMemset(ptr, 0, size));
#else
  RUNTIME_ERROR("do not have CUDA");
#endif
}

Device::Device(int device_id, DeviceType device_type): device_id_(device_id), device_type_(device_type) {
  if (DeviceType::CPU == device_type_) {
    // CPU device is -1
    device_id_ = -1;
    allocator_ = std::make_shared<CPUAllocator>();
  } else if (DeviceType::GPU == device_type_) {
    ARGUMENT_CHECK(device_id_ >= 0, "For GPU device, device_id must >= 0.")
    allocator_ = std::make_shared<GPUAllocator>(device_id_);
  } else {
    RUNTIME_ERROR("the device type is not support");
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
void* Device::malloc(size_t size) {
  return allocator_->malloc(size);
}

// free the memory
void Device::free(void* ptr) {
  allocator_->free(ptr);
}

// zero the memory
void Device::zero(void *ptr, size_t size) {
  allocator_->zero(ptr, size);
}

SpinLock device_locker_;
std::unordered_map<int, std::shared_ptr<Device>> devices_;

std::shared_ptr<Device> get_device(int device_id) {
  SpinLockHandler handler(device_locker_);

  if (device_id < 0) {
    device_id = -1;
  }

  if (devices_.find(device_id) == devices_.end()) {
    if (device_id < 0) {
      devices_[device_id] = std::make_shared<Device>(device_id, DeviceType::CPU);
    } else {
      devices_[device_id] = std::make_shared<Device>(device_id, DeviceType::GPU);
    }
  }

  return devices_[device_id];
}

std::shared_ptr<Device> get_cpu_device() {
  return get_device(-1);
}

std::shared_ptr<Device> get_gpu_device(int device_id) {
  ARGUMENT_CHECK(device_id >= 0, "GPU device's id must >= 0")
  return get_device(device_id);
}

}
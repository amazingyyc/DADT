
#ifdef HAVE_CUDA
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#endif

#include <iostream>
#include <unordered_map>

#include "exception.h"
#include "device.h"

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

// copy memory from cpu
void CPUAllocator::memcpy_from_cpu(void *dst, const void *src, size_t size) {
  std::memcpy(dst, src, size);
}

// copy from GPU
void CPUAllocator::memcpy_from_gpu(void *dst, const void *src, size_t size) {
#ifdef HAVE_CUDA
  CUDA_CALL(cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost));
#else
  RUNTIME_ERROR("do not have CUDA");
#endif
}

// copy memory to cpu
void CPUAllocator::memcpy_to_cpu(void *dst, const void *src, size_t size) {
  std::memcpy(dst, src, size);
}

// copy memory to gpu
void CPUAllocator::memcpy_to_gpu(void *dst, const void *src, size_t size) {
#ifdef HAVE_CUDA
  CUDA_CALL(cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice));
#else
  RUNTIME_ERROR("do not have CUDA");
#endif
}

GPUAllocator::GPUAllocator(int device_id): device_id_(device_id) {
}

void* GPUAllocator::malloc(size_t size) {
#ifdef HAVE_CUDA
  void *ptr;

  CUDA_CALL(cudaSetDevice(device_id_));
  CUDA_CALL(cudaMalloc(&ptr, size));

  return ptr;
#else
  RUNTIME_ERROR("do not have CUDA");
#endif
}

void GPUAllocator::free(void *ptr) {
#ifdef HAVE_CUDA
  CUDA_CALL(cudaSetDevice(device_id_));
  CUDA_CALL(cudaFree(ptr));
#else
  RUNTIME_ERROR("do not have CUDA");
#endif
}

// zero the memory
void GPUAllocator::zero(void *ptr, size_t size) {
  #ifdef HAVE_CUDA
  CUDA_CALL(cudaSetDevice(device_id_));
  CUDA_CALL(cudaMemset(ptr, 0, size));
#else
  RUNTIME_ERROR("do not have CUDA");
#endif
}

// copy memory from cpu
void GPUAllocator::memcpy_from_cpu(void *dst, const void *src, size_t size) {
#ifdef HAVE_CUDA
  CUDA_CALL(cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice));
#else
  RUNTIME_ERROR("do not have CUDA");
#endif
}

// copy from GPU
void GPUAllocator::memcpy_from_gpu(void *dst, const void *src, size_t size) {
#ifdef HAVE_CUDA
  CUDA_CALL(cudaMemcpy(dst, src, size, cudaMemcpyDeviceToDevice));
#else
  RUNTIME_ERROR("do not have CUDA");
#endif
}

// copy memory to cpu
void GPUAllocator::memcpy_to_cpu(void *dst, const void *src, size_t size) {
#ifdef HAVE_CUDA
  CUDA_CALL(cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost));
#else
  RUNTIME_ERROR("do not have CUDA");
#endif
}

// copy memory to gpu
void GPUAllocator::memcpy_to_gpu(void *dst, const void *src, size_t size) {
#ifdef HAVE_CUDA
  CUDA_CALL(cudaMemcpy(dst, src, size, cudaMemcpyDeviceToDevice));
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

// copy memory from cpu
void Device::memcpy_from_cpu(void *dst, const void *src, size_t size) {
  allocator_->memcpy_from_cpu(dst, src, size);
}

// copy from GPU
void Device::memcpy_from_gpu(void *dst, const void *src, size_t size) {
  allocator_->memcpy_from_gpu(dst, src, size);
}

// copy memory to cpu
void Device::memcpy_to_cpu(void *dst, const void *src, size_t size) {
  allocator_->memcpy_to_cpu(dst, src, size);
}

// copy memory to gpu
void Device::memcpy_to_gpu(void *dst, const void *src, size_t size) {
  allocator_->memcpy_to_gpu(dst, src, size);
}

std::shared_ptr<Device> cpu_device_;
std::unordered_map<int, std::shared_ptr<Device>> gpu_device_;

std::shared_ptr<Device> get_cpu_device() {
  if (nullptr == cpu_device_) {
    cpu_device_ = std::make_shared<Device>(-1, DeviceType::CPU);
  }

  return cpu_device_;
}

std::shared_ptr<Device> get_gpu_device(int device_id) {
  if (gpu_device_.find(device_id) == gpu_device_.end()) {
    gpu_device_[device_id] = std::make_shared<Device>(device_id, DeviceType::GPU);
  }

  return gpu_device_[device_id];
}

}
#include "tensor.h"

namespace dadt {

Tensor::Tensor(std::shared_ptr<TensorStorage> storage, size_t offset, Shape shape, ElementType element_type)
:storage_(storage), offset_(offset), shape_(shape), element_type_(element_type) {
}

std::shared_ptr<Device> Tensor::device() {
  return storage_->device();
}

size_t Tensor::offset() const {
  return offset_;
}

const Shape &Tensor::shape() const {
  return shape_;
}

const ElementType &Tensor::element_type() const {
  return element_type_;
}

void *Tensor::ptr() {
  return ((uint8_t *)storage_->ptr()) + offset_;
}

void *Tensor::ptr() const {
  return ((uint8_t *)storage_->ptr()) + offset_;
}

bool Tensor::is_scalar() const {
  return shape_.is_scalar();
}

int Tensor::size() const {
  return shape_.size();
}

int Tensor::num_bytes() const {
  return element_type_.byte_width() * size();
}

int Tensor::dim(int axis) const {
  return shape_.dim(axis);
}

// copy memory from other 
// for GPU the copy is synchronous 
void Tensor::copy_from(const void *data, bool is_gpu) {
  if (is_gpu) {
#ifdef HAVE_NCCL
    if (DeviceType::CPU == device()->device_type()) {
      // copy from GPU to CPU
      CUDA_CALL(cudaMemcpy(ptr(), data, num_bytes(), cudaMemcpyDeviceToHost));
    } else {
      // copy from gpu to gpu
      CUDA_CALL(cudaMemcpy(ptr(), data, num_bytes(), cudaMemcpyDeviceToDevice));
    }
#else
      RUNTIME_ERROR("compile without CUDA, can not call CUDA function");
#endif
  } else {
    if (DeviceType::CPU == device()->device_type()) {
      // cpu to cpu
      std::memcpy(ptr(), data, num_bytes());
    } else {
#ifdef HAVE_NCCL
      // cpu to gpu
      CUDA_CALL(cudaMemcpy(ptr(), data, num_bytes(), cudaMemcpyHostToDevice));
#else
      RUNTIME_ERROR("compile without CUDA, can not call CUDA function");
#endif
    }
  }
}

// copy memory to other
// synchronous 
void Tensor::copy_to(void *data, bool is_gpu) {
  if (is_gpu) {
#ifdef HAVE_NCCL
    if (DeviceType::CPU == device()->device_type()) {
      // from cpu to gpu
      CUDA_CALL(cudaMemcpy(data, ptr(), num_bytes(), cudaMemcpyHostToDevice));
    } else {
      // from gpu to gpu
      CUDA_CALL(cudaMemcpy(data, ptr(), num_bytes(), cudaMemcpyDeviceToDevice));
    }
#else
      RUNTIME_ERROR("compile without CUDA, can not call CUDA function");
#endif
  } else {
    if (DeviceType::CPU == device()->device_type()) {
      // cpu to cpu
      std::memcpy(data, ptr(), num_bytes());
    } else {
      // copy memory from gpu to cpu
#ifdef HAVE_NCCL
      CUDA_CALL(cudaMemcpy(dadt, ptr(), num_bytes(), cudaMemcpyDeviceToHost));
#else
      RUNTIME_ERROR("compile without CUDA, can not call CUDA function");
#endif
    }
  }
}

}
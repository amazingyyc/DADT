#include "pytorch_tensor_impl.h"

#ifdef HAVE_NCCL
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <nccl.h>
#endif

#include "common/exception.h"
#include "pytorch_utils.h"

namespace dadt {
namespace pytorch {

PytorchTensorImpl::PytorchTensorImpl(const torch::Tensor& torch_tensor)
    : torch_tensor_(torch_tensor) {
  ARGUMENT_CHECK(torch_tensor_.layout() == torch::kStrided,
                 "PytorchTensorImpl need DenseTensor");
}

const torch::Tensor& PytorchTensorImpl::torch_tensor() const {
  return torch_tensor_;
}

bool PytorchTensorImpl::IsDense() const {
  return torch_tensor_.layout() == torch::kStrided;
}

bool PytorchTensorImpl::IsCoo() const {
  return torch_tensor_.layout() == torch::kSparse;
}

bool PytorchTensorImpl::IsCpu() const {
  // Not call is_cpu(), Compitable with pytorch old version.
  return torch_tensor_.device().type() == at::DeviceType::CPU;
}

bool PytorchTensorImpl::IsCuda() const {
  return torch_tensor_.is_cuda();
}

int PytorchTensorImpl::DeviceId() const {
  if (torch_tensor_.is_cuda()) {
    return torch_tensor_.device().index();
  }

  return -1;
}

bool PytorchTensorImpl::IsContiguous() const {
  return torch_tensor_.is_contiguous();
}

ElementType PytorchTensorImpl::element_type() const {
  return TorchDTypeToElementType(torch_tensor_.scalar_type());
}

Shape PytorchTensorImpl::shape() const {
  return TorchSizesToShape(torch_tensor_.sizes());
}

int64_t PytorchTensorImpl::Size() const {
  return torch_tensor_.numel();
}

size_t PytorchTensorImpl::NumBytes() const {
  return torch_tensor_.nbytes();
}

void* PytorchTensorImpl::Ptr() {
  return torch_tensor_.data_ptr();
}

void* PytorchTensorImpl::Ptr() const {
  return torch_tensor_.data_ptr();
}

std::shared_ptr<TensorImpl> PytorchTensorImpl::DynamicZero(
    const Shape& shape, ElementType element_type) const {
  auto options = torch::TensorOptions()
                     .dtype(ElementTypeToTorchDType(element_type))
                     .layout(torch::kStrided)
                     .device(torch_tensor_.device());
  torch::IntArrayRef sizes(shape.dims());
  torch::Tensor tensor = torch::zeros(sizes, options);

  if (tensor.is_cuda()) {
#ifdef HAVE_NCCL
    // Cuda is async we need to wait the Cuda operate finish
    cudaStream_t cuda_stream =
        at::cuda::getCurrentCUDAStream(tensor.device().index()).stream();
    CUDA_CALL(cudaStreamSynchronize(cuda_stream));
#else
    RUNTIME_ERROR("Build without Cuda");
#endif
  }

  return std::make_shared<PytorchTensorImpl>(tensor);
}

}  // namespace pytorch
}  // namespace dadt

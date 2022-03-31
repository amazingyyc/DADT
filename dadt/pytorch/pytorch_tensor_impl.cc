#include "pytorch_tensor_impl.h"

#ifdef HAVE_NCCL
#include "pytorch_cuda_stream_guard.h"
#endif

#include "pytorch_utils.h"

namespace dadt {
namespace pytorch {

PytorchTensorImpl::PytorchTensorImpl(torch::Tensor torch_tensor)
    : torch_tensor_(torch_tensor) {
}

torch::Tensor PytorchTensorImpl::torch_tensor() const {
  return torch_tensor_;
}

bool PytorchTensorImpl::IsCoo() const {
  return torch_tensor_.layout() == torch::kSparse;
}

std::shared_ptr<TensorImpl> PytorchTensorImpl::Indices() const {
  return std::make_shared<PytorchTensorImpl>(torch_tensor_.indices());
}

std::shared_ptr<TensorImpl> PytorchTensorImpl::Values() const {
  return std::make_shared<PytorchTensorImpl>(torch_tensor_.values());
}

int64_t PytorchTensorImpl::SparseDim() const {
  return torch_tensor_.sparse_dim();
}

int64_t PytorchTensorImpl::DenseDim() const {
  return torch_tensor_.dense_dim();
}

int64_t PytorchTensorImpl::nnz() const {
  return torch_tensor_._nnz();
}

bool PytorchTensorImpl::IsCoalesced() const {
  return torch_tensor_.is_coalesced();
}

bool PytorchTensorImpl::IsDense() const {
  return torch_tensor_.layout() == torch::kStrided;
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

bool PytorchTensorImpl::IsCpu() const {
  return torch_tensor_.is_cpu();
}

bool PytorchTensorImpl::IsCuda() const {
  return torch_tensor_.is_cuda();
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

std::shared_ptr<TensorImpl> PytorchTensorImpl::Transpose(int64_t dim0,
                                                         int64_t dim1) const {
  auto tensor_t = torch_tensor_.transpose(dim0, dim1);

  return std::make_shared<PytorchTensorImpl>(tensor_t);
}

std::shared_ptr<TensorImpl> PytorchTensorImpl::Coalesce() const {
  auto new_tensor = torch_tensor_.coalesce();

  return std::make_shared<PytorchTensorImpl>(new_tensor);
}

#ifdef HAVE_NCCL
// Return a cuda stream guard.
std::unique_ptr<StreamGuard> PytorchTensorImpl::DynamicCudaStreamGuard(
    cudaStream_t cuda_stream, int8_t device_index) const {
  return std::unique_ptr<PytorchCudaStreamGuard>(
      new PytorchCudaStreamGuard(cuda_stream, device_index));
}
#endif

std::shared_ptr<TensorImpl> PytorchTensorImpl::DynamicZero(
    const Shape& shape, ElementType element_type) const {
  auto options = torch::TensorOptions()
                     .dtype(ElementTypeToTorchDType(element_type))
                     .layout(torch::kStrided)
                     .device(torch_tensor_.device());

  auto sizes = ShapeToTorchSizes(shape);

  return std::make_shared<PytorchTensorImpl>(torch::zeros(sizes, options));
}

std::shared_ptr<TensorImpl> PytorchTensorImpl::DynamicCoo(
    std::shared_ptr<TensorImpl> indices, std::shared_ptr<TensorImpl> values,
    const Shape& shape) const {
  auto* indices_p = dynamic_cast<PytorchTensorImpl*>(indices.get());
  auto* values_p = dynamic_cast<PytorchTensorImpl*>(values.get());

  torch::Tensor coo_tensor = torch::sparse_coo_tensor(indices_p->torch_tensor_,
                                                      values_p->torch_tensor_,
                                                      ShapeToTorchSizes(shape));

  return std::make_shared<PytorchTensorImpl>(coo_tensor);
}

}  // namespace pytorch
}  // namespace dadt

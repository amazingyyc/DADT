#include "tensorflow_utils.h"

// if the OpKernelContext is GPU
bool is_gpu_conext(OpKernelContext* context) {
  if (nullptr != context->device() &&
      nullptr != context->device()->tensorflow_gpu_device_info()) {
    return true;
  }

  return false;
}
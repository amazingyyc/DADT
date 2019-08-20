#ifndef TENSORFLOW_UTILS_H
#define TENSORFLOW_UTILS_H

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"

// if the OpKernelContext is GPU
bool is_gpu_conext(OpKernelContext* context);

#endif
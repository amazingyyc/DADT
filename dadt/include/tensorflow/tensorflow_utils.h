#ifndef TENSORFLOW_UTILS_H
#define TENSORFLOW_UTILS_H

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"

#include "internal.h"

// if the OpKernelContext is GPU
bool is_gpu_conext(OpKernelContext* context);

ElementType convert_dtype_to_element_type(DataType dtype);

std::vector<int> convert_tensor_shape_to_array(const TensorShape& shape);

#endif
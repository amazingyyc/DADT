#ifndef PYTORCH_UTILS_H
#define PYTORCH_UTILS_H

#include <torch/extension.h>
#include <torch/torch.h>

#include "lock_tensor.h"

namespace dadt {
namespace pytorch {

dadt::ElementType get_element_type(const torch::Tensor &x);

dadt::Shape get_shape_vector(const torch::Tensor &x);

dadt::ElementType parse_element_type(const torch::Tensor &x);

dadt::Shape parse_shape_vector(const torch::Tensor &x);

}
}

#endif
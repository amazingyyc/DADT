#ifndef TENSOR_H
#define TENSOR_H

#include <iostream>

#include "shape.h"
#include "element_type.h"
#include "tensor_storage.h"

namespace dadt {

class Tensor {
protected:
  // memory
  std::shared_ptr<TensorStorage> storage_;

  // offset of tensor
  size_t offset_;

  // the tensor shape
  Shape shape_;

  // element type
  ElementType element_type_;

public:
  explicit Tensor(std::shared_ptr<TensorStorage> storage, size_t offset, Shape shape, ElementType type);

  std::shared_ptr<Device> device();

  size_t offset() const;

  const Shape &shape() const;

  const ElementType &element_type() const;

  void *ptr();
  void *ptr() const;

  template <typename T> T* data() { 
    return (T*) ptr(); 
  }

  template <typename T> T* data() const { 
    return (T*) ptr(); 
  }

  bool is_scalar() const;

  int size() const;

  int num_bytes() const;

  int dim(int) const;
};

}

#endif
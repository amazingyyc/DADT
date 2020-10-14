#ifndef LOCK_TENSOR_H
#define LOCK_TENSOR_H

#include <iostream>
#include <string>
#include <memory>
#include <cstring>

#include "device.h"
#include "task.h"
#include "spin_lock.h"
#include "shape.h"
#include "element_type.h"
#include "tensor_storage.h"

namespace dadt {


// the lock tensor is a tensor add a spin lock
// at every different status the tensor should do different operator
// 
// the lock tensor status transition steps
// kInExecute: after fill new data, the tensor will put in a queue to do task, at here the status become kInExecute
// kWaitForFetch: alter reduce thread exhcange data cross all node the tensor status will become kWaitForFetch
// kInFetch: the compute thread is fetch data from the tensor and put new data in to it
// alter finish fetch than status become kInExecute
enum class LockTensorStatus: int {
  kInExecute    = 0,
  kWaitForFetch = 1,
  kInFetch      = 2,
};


class LockTensor {
protected:
  // unique name
  std::string name_;

  // represent the tesnor status
  SpinStatusLock status_;

  // memory
  std::shared_ptr<TensorStorage> storage_;

  // offset of tensor
  size_t offset_;

  // the tensor shape
  Shape shape_;

  // element type
  ElementType element_type_;

public:
  LockTensor(
    std::shared_ptr<TensorStorage> storage,
    size_t offset,
    Shape shape,
    ElementType element_type,
    std::string name,
    LockTensorStatus initialize_status);

  // wait the tensor is expected_status/new_status and change it to new_status
  void wait(LockTensorStatus expected_status, LockTensorStatus new_status);

  // get name
  const std::string& name() const;

  const Shape& shape() const;

  const ElementType& element_type() const;

  bool is_scalar() const;
  int size() const;
  int num_bytes() const;
  int dim(int) const;

  bool is_cpu() const;

  // return GPU device id is CPU return -1
  virtual int device_id() const;

  // whether is a cuda tensor
  virtual bool is_cuda() const;

  // get memory pointer
  virtual void* ptr();
  virtual void* ptr() const;

};

}

#endif
#ifndef LOCK_TENSOR_H
#define LOCK_TENSOR_H

#include <iostream>
#include <string>
#include <memory>
#include <cstring>

#include "device.h"
#include "task.h"
#include "tensor.h"
#include "spin_lock.h"

namespace dadt {


// the lock tensor is a tensor add a spin lock
// at every different status the tensor should do different operator
// 
// the lock tensor status transition steps
// InExecute: after fill new data, the tensor will put in a queue to do task, at here the status become InExecute
// WaitForFetch: alter reduce thread exhcange data cross all node the tensor status will become WaitForFetch
// InFetch: the compute thread is fetch data from the tensor and put new data in to it
// alter finish fetch than status become InExecute
enum class LockTensorStatus: int {
  InExecute    = 1,
  WaitForFetch = 2,
  InFetch      = 3,
};

class LockTensor: public Tensor {
private:
  // unique name
  std::string name_;

  // represent the tesnor status
  SpinLock status_;

public:
  LockTensor(std::shared_ptr<TensorStorage> storage, 
            size_t offset, 
            Shape shape, 
            ElementType type,
            std::string name, 
            LockTensorStatus initialize_status);

  std::string name();

  // wait the tensor is expected_status/new_status and change it to new_status
  void wait(LockTensorStatus expected_status, LockTensorStatus new_status);
};

}

#endif
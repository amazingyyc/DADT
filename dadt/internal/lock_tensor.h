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

/**
 * the lock tensor is a tensor add a spin lock
 * at every different status the tensor should do different operator
 * 
 * the  lock tensor status transition steps
 * InFill: when compute thread begin to fill data, than the status become InFill
 * InAllReduce: alter fill new data, the tensor will put in a queue to do all reduce, at here the status become InAllReduce
 * WaitForFetch: alter reduce thread exhcange data cross all node the tensor status will become WaitForFetch
 * InFetch: the compute thread is fetch data from the tensor
 * alter finish fetch than status become InFill
 */
enum class LockTensorStatus: int {
  InFill       = 1, // the compute thread is filling data to this tensor
  InExecute    = 2, // after fill new data than the tensor will be in put it a queue than will do allreduce or broadcast
  WaitForFetch = 3, //after all reduce the tensor will wait the compute to fetch back
  InFetch      = 4, // the compute thread is fetch the data back in fetch
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
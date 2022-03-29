#pragma once

#include <memory>
#include <string>

#include "common/spin_lock.h"
#include "t/element_type.h"
#include "t/tensor.h"

namespace dadt {

// the lock tensor is a tensor add a spin lock
// at every different status the tensor should do different operator
//
// the lock tensor status transition steps
// kInExecute: after fill new data, the tensor will put in a queue to do task,
// at here the status become kInExecute kWaitForFetch: alter reduce thread
// exhcange data cross all node the tensor status will become kWaitForFetch
// kInFetch: the compute thread is fetch data from the tensor and put new data
// in to it alter finish fetch than status become kInExecute
enum struct LockTensorStatus : uint32_t {
  kInExecute = 0,
  kWaitForFetch = 1,
  kInFetch = 2,
};

class LockTensor {
private:
  // represent the tesnor status
  SpinStatusLock status_;

  Tensor tensor_;

public:
  LockTensor(LockTensorStatus init_status, Tensor tensor);

  ~LockTensor() = default;

  // wait the tensor is expected_status/new_status and change it to new_status
  void Wait(LockTensorStatus expected_status, LockTensorStatus new_status);

  const Tensor& tensor() const;

  void ResetTensor(const Tensor& tensor);
};

}  // namespace dadt

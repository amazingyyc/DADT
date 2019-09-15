#ifndef MPI_ALL_REDUCE_EXECUTOR_H
#define MPI_ALL_REDUCE_EXECUTOR_H

#include <iostream>
#include <unordered_map>
#include  <mutex>

#include "device.h"
#include "task_executor.h"
#include "memory_buffer.h"

namespace dadt {

class MPIAllReduceExecutor: public ITaskExecutor {
private:
  // tensor_pool_ will used in multi-thread
  std::mutex pool_mutex_;

  // allreduce will reuse the tensor, so use a map to store it
  std::unordered_map<std::string, std::shared_ptr<LockTensor>> tensor_pool_;

  MemoryBuffer buffer_;

public:
  MPIAllReduceExecutor(size_t buffer_size = 67108864);

  // whether already create a midway tensor
  std::shared_ptr<LockTensor> obtain_midway_tensor(std::string name) override;
  
  std::shared_ptr<LockTensor> create_midway_tensor(std::string name, std::vector<int> dims, ElementType element_type) override;

  void operator()(const Context &context, const std::vector<Task> &tasks, std::shared_ptr<TimeLine> timeline) override;
};

}

#endif

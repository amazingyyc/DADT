#pragma once

#include <iostream>
#include <thread>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "common/context.h"
#include "common/task.h"
#include "communicator.h"
#include "concurrentqueue.h"
#include "executor/task_executor.h"

namespace dadt {

// commander of whole project
class Commander {
private:
  // context
  Context context_;

  // a flag represent whether dadt finish initizlized
  std::atomic<bool> initialized_;

  // a map store the task_type with corresponding task executor
  std::unordered_map<TaskType, std::unique_ptr<ITaskExecutor>> task_executors_;

  moodycamel::ConcurrentQueue<Task> task_queue_;

  std::thread worker_thread_;

  // communicator
  Communicator communicator_;

  // LockTensor cache.
  std::unordered_map<TaskKey, std::shared_ptr<LockTensor>, TaskKeyHash,
                     TaskKeyEqual>
      cached_l_tensors_;

private:
  // Setup the commander
  void Setup(const Config& config);

  // clean context, call in the same thread with init_context
  void Clear();

  // loop to get task from queue and do the task
  // return wheter shut down
  bool DoTask();

  // used for background_thread_ to do the task
  void Run(const Config& config);

public:
  Commander();

  // init the commander
  void Initialize(const Config& config);

  // shutdown background thread
  void Shutdown();

  // if the commander have been initialized
  bool Initialized() const;

  // process count
  int Size() const;

  // local machine process count
  int LocalSize() const;

  // process rank
  int Rank() const;

  // local rank
  int LocalRank() const;

  // barrier all process
  void Barrier();

  // local barrier process
  void LocalBarrier();

  std::shared_ptr<LockTensor> CachedLTensor(TaskKey key) const;

  void InsertLTensor(TaskKey key, std::shared_ptr<LockTensor> l_tensor);

  // insert a task to task queue
  void EnqueueTask(Task&& t);
};

}  // namespace dadt

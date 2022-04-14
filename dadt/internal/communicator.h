#pragma once

#include <unordered_map>
#include <vector>

#include "common/buffer.h"
#include "common/context.h"
#include "common/task.h"
#include "concurrentqueue.h"

namespace dadt {

class Communicator {
private:
  // A map store the Task that ready in current rank.
  std::unordered_map<TaskKey, Task, TaskKeyHash, TaskKeyEqual>
      waiting_task_pool_;

private:
  void ExchangeTaskKeys(const Context& context,
                        const std::vector<TaskKey>& task_keys,
                        std::vector<TaskKey>* total_task_keys);

public:
  Communicator();

  ~Communicator() = default;

  // at here will exchange with other rank get ready task
  std::unordered_map<TaskType, std::vector<Task>> Exchange(
      const Context& context, moodycamel::ConcurrentQueue<Task>& task_queue);
};

}  // namespace dadt

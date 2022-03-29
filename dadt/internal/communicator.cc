#include "communicator.h"

#include <mpi.h>

#include <memory>

#include "common/context.h"
#include "common/deserialize.h"
#include "common/device.h"
#include "common/exception.h"
#include "common/mem_buffer.h"
#include "common/mem_reader.h"
#include "common/mem_writer.h"
#include "common/serialize.h"
#include "t/lock_tensor.h"

namespace dadt {

Communicator::Communicator() {
}

void Communicator::ExchangeTaskKeys(const Context& context,
                                    const std::vector<TaskKey>& task_keys,
                                    std::vector<TaskKey>* total_task_keys) {
  // serialize to bytes.
  MemWriter writer;
  Serialize serialize(&writer);

  ARGUMENT_CHECK(serialize << task_keys, "Serialize vector of TaskKey error!");

  // Exchange the memory bytes size.
  std::vector<int> mem_lengths(context.world_size);
  mem_lengths[context.world_rank] = (int)writer.offset();

  MPI_CALL(MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, mem_lengths.data(),
                         1, MPI_INT, context.world_comm));

  int total_mem_length = 0;
  for (auto i : mem_lengths) {
    total_mem_length += i;
  }

  MemBuffer mem_buffer(total_mem_length);
  std::vector<int> recvcounts(context.world_size);
  std::vector<int> displs(context.world_size);

  for (int i = 0; i < context.world_size; ++i) {
    recvcounts[i] = mem_lengths[i];

    if (0 == i) {
      displs[i] = 0;
    } else {
      displs[i] = displs[i - 1] + recvcounts[i - 1];
    }
  }

  MPI_CALL(MPI_Allgatherv(writer.ptr(), (int)writer.offset(), MPI_CHAR,
                          mem_buffer.ptr(), recvcounts.data(), displs.data(),
                          MPI_CHAR, context.world_comm));

  total_task_keys->clear();
  for (int i = 0; i < context.world_size; ++i) {
    MemReader reader((const char*)mem_buffer.ptr(displs[i]), recvcounts[i]);
    Deserialize deserialize(&reader);

    std::vector<TaskKey> t_task_keys;

    ARGUMENT_CHECK(deserialize >> t_task_keys, "Deserialize TaskKeys error!");

    total_task_keys->insert(total_task_keys->end(), t_task_keys.begin(),
                            t_task_keys.end());
  }
}

// at here will exchange with other rank get ready task
std::unordered_map<TaskType, std::vector<Task>> Communicator::Exchange(
    const Context& context, moodycamel::ConcurrentQueue<Task>& task_queue) {
  std::vector<Task> ready_tasks;

  // dequeue task from queue
  while (true) {
    Task t;
    if (task_queue.try_dequeue(t)) {
      ready_tasks.emplace_back(std::move(t));
    } else {
      break;
    }
  }

  // firstly put ready_task into waiting_task_pool_
  for (auto& task : ready_tasks) {
    TaskKey key;
    key.type = task.type;
    key.id = task.id;

    waiting_task_pool_.emplace(key, std::move(task));
  }

  std::vector<TaskKey> ready_keys;
  ready_keys.reserve(waiting_task_pool_.size());
  for (const auto& [key, _] : waiting_task_pool_) {
    ready_keys.emplace_back(key);
  }

  std::vector<TaskKey> total_ready_keys;
  ExchangeTaskKeys(context, ready_keys, &total_ready_keys);

  std::unordered_map<TaskType, std::vector<Task>> should_execute_tasks;

  // For now every rank has the same ready taskey.
  // Than we need check which task key is ready for all rank.
  std::unordered_map<TaskKey, size_t, TaskKeyHash, TaskKeyEqual> counts;
  for (const auto& key : total_ready_keys) {
    counts[key]++;

    if (counts[key] == context.world_size) {
      // The Task is read for all rank.
      auto it = waiting_task_pool_.find(key);
      ARGUMENT_CHECK(it != waiting_task_pool_.end(), "Can't find TaskKey.");

      should_execute_tasks[key.type].emplace_back(std::move(it->second));
      waiting_task_pool_.erase(it);
    }
  }

  return should_execute_tasks;
}

}  // namespace dadt

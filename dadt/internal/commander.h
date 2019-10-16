#ifndef COMMANDER_H
#define COMMANDER_H

#include <iostream>
#include <thread>
#include <vector>
#include <unordered_set>
#include <unordered_map>
#include <tuple>

#include "concurrentqueue.h"
#include "task.h"
#include "context.h"
#include "task_executor.h"
#include "thread_pool.h"
#include "communicator.h"
#include "timeline.h"

namespace dadt {

// commander of whole project
class Commander {
private:
  // context
  Context context_;

  // a flag represent whether dadt finish initizlized
  std::atomic<bool> initialized_;
  // a map store the task_type with corresponding task executor
  std::unordered_map<TaskType, std::shared_ptr<ITaskExecutor>> task_executors_;

  // a task queue, will accept the message and a background will get the message and do the task
  moodycamel::ConcurrentQueue<Task> task_queue_;

  // a background thread will loop take task from message_queue_ to do the task
  std::thread worker_thread_;

  // communicator
  Communicator communicator_;

  // timeline
  std::shared_ptr<TimeLine> timeline_;

#ifdef HAVE_NCCL
  // every thread has a unique cuda event
  // used by op to wait cuda kernel finish
  std::mutex op_cuda_event_mutex_;
  std::unordered_map<std::thread::id, cudaEvent_t> op_cuda_events_;
#endif

private:
  // initialize context
  void init_context(Config);

  // clean context, call in the same thread with init_context
  void clear_context();

  // loop to get task from queue and do the task
  // return wheter shut down
  bool worker_do_task();

public:
  Commander();

  // init the commander
  void init(Config config);

  // shutdown background thread
  void shutdown();

  // if the commander have been initialized
  bool initialized();

  // process count
  int size();

  // local machine process count
  int local_size();

  // process rank
  int rank();

  // local rank
  int local_rank();

  // barrier all process
  void barrier();

  // local barrier process
  void local_barrier();

  // insert a task to task queue
  void enqueue_task(Task &&t);

  // begin timeline evet
  void begin_timeline_event(const std::string &name, const std::string &event);

  // end timeline event
  void end_timeline_event(const std::string &name, const std::string &event);

#ifdef HAVE_NCCL
  // every thread has a unique cuda event
  cudaEvent_t obtain_cuda_event();
#endif

  // check whether already create a midway tesnor
  // it is thread safe
  std::shared_ptr<LockTensor> obtain_midway_tensor(TaskType task_type, std::string name);

  // get a interim tensor by TaskType
  // it is thread safe
  std::shared_ptr<LockTensor> create_midway_tensor(TaskType task_type, std::string name, std::vector<int> dims, ElementType element_type);
  
  // used for background_thread_ to do the task
  void worker_do_cycle(Config config);
};

}

#endif
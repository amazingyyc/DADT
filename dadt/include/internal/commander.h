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

namespace dadt {

/**
 * commander of all project
 */
class Commander {
private:
  // context
  Context context_;

  // a map store the task_type with corresponding task op
  std::unordered_map<TaskType, std::shared_ptr<ITaskExecutor>> task_executors_;

  // a flag represent if dadt finish initizlized
  std::atomic<bool> initialized_;

  // a task queue, will accept the message and a background will get the message and do the task
  moodycamel::ConcurrentQueue<Task> task_queue_;

  // a background thread will loop take task from message_queue_ to do the task
  std::thread worker_thread_;

  // if stop the background thread
  std::atomic<bool> worker_stopped_;

  // after take the task from queue, it may not execute right now becuase some other process may not put same some task
  // so put it in register for tmp
  std::unordered_map<TaskKey, Task, TaskKeyHash, TaskKeyEqual> task_pool_;

  // this map used for record how many process has put same task in queue
  std::unordered_map<TaskKey, std::unordered_set<int>, TaskKeyHash, TaskKeyEqual> task_register_;

  // for deep learning framwork like tensorflow
  // we will use async op to do the braodcast and allreduce
  // so use a thread pool to do the op
  ThreadPool async_queue_;

private:
  // initialize context
  void init_context();

  // exchange string with special MPI_Comm
  // return is a string array corresponding the rank index
  std::vector<std::string> exchange_string(MPI_Comm mpi_comm, int rank, int size, std::string &str); 

  // formate a category task to json str
  std::string dump_tasks_to_json(const std::unordered_map<TaskType, std::vector<std::string>> &category_tasks);

  // parse task from json str
  std::unordered_map<TaskType, std::vector<std::string>> parse_json_to_tasks(const std::string &json_str);

  // insert a ready tensor and decide if it it ready to execute
  // only if all rank has ready to do the task
  bool if_execute_task(int rank, TaskType task_type, std::string name);

  // exchange the tasks with each process
  std::unordered_map<TaskType, std::vector<Task>> exchange_execute_tasks(std::vector<Task> &tasks);

  // loop to get task from queue and do the task
  void worker_do_task();

public:
  Commander();

  // init the commander
  void init();

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

  // used for background_thread_ to do the task
  void worker_do_cycle();

  // stop the back ground thread
  void stop_worker();
  
  std::shared_ptr<LockTensor> has_midway_tensor(TaskType task_type, std::string name);

  // get a interim tensor by TaskType
  std::shared_ptr<LockTensor> midway_tensor(TaskType task_type, 
                                            std::string name, 
                                            std::vector<int> dims, 
                                            ElementType element_type);

  // put a task is async queue
  void async_job(std::function<void()> &&task);
};

}

#endif
#ifndef COMMUNICATOR_H
#define COMMUNICATOR_H

#include <iostream>
#include <unordered_map>

#include "concurrentqueue.h"
#include "task.h"
#include "group.h"

namespace dadt {

// a cell include task type, name, and corresponding tesnor size
struct TaskCell {
  TaskType task_type;

  std::string name;

  int num_bytes;
};

// the communicator will impellemt bitAllReduce and Group ref paper: EXASCALE DEEP LEARNING FOR SCIENTIFIC INVERSE PROBLEMS
class Communicator {
private:
  // use a vector contain the task cell, sort by dequeue from queue
  std::vector<TaskCell> task_cells_;

  // every task will assign a id
  std::unordered_map<TaskKey, int32_t, TaskKeyHash, TaskKeyEqual> task_id_;

  // id-task map
  std::unordered_map<int32_t, TaskKey> id_task_;

  // the taskkey will assign to different group
  std::vector<std::shared_ptr<Group>> groups_;

  // task-group map
  std::unordered_map<TaskKey, std::shared_ptr<Group>, TaskKeyHash, TaskKeyEqual> task_group_;

  // wait for Group task
  // when get task from queue the task will put into waiting_group_pool_
  // than get task from waiting_group_pool_ to check whether all task from one group has ready.
  // if yes, put the ready task into waiting_request_pool_, waiting_request_pool_ means the task has ready in this rank
  // should wait other rank ready
  std::unordered_map<TaskKey, Task, TaskKeyHash, TaskKeyEqual> waiting_group_pool_;

  // the task in this pool means already pass group check, waiting other rank ready
  std::unordered_map<TaskKey, Task, TaskKeyHash, TaskKeyEqual> waiting_request_pool_;

private:
  std::vector<std::string> exchange_string(MPI_Comm mpi_comm, int rank, int size, std::string &str);

  // format tskcell to string
  std::string dump_tasks_cells_to_json(const std::vector<TaskCell> &task_cells);

  // parse task cell from json
  std::vector<TaskCell> parse_json_to_task_cells(const std::string &json_str);

  // update group and task cell
  void update(Context context, const vector<Task> &ready_task);

public:

  // at here will exchange with other rank get ready task
  std::unordered_map<TaskType, std::vector<Task>> exchange(const Context &contxt, 
                                                           const moodycamel::ConcurrentQueue<Task> task_queue);
};

}

#endif
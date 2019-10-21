#ifndef COMMUNICATOR_H
#define COMMUNICATOR_H

#include <iostream>
#include <unordered_map>
#include <vector>

#include "concurrentqueue.h"
#include "types.h"
#include "task.h"
#include "group.h"

namespace dadt {

// the communicator will impellemt bitAllReduce and Group, ref paper: EXASCALE DEEP LEARNING FOR SCIENTIFIC INVERSE PROBLEMS
class Communicator {
private:
  // use a vector contain the task cell
  // a taskcell include a num_bytes used for grouping
  std::vector<TaskCell> task_cells_;

  // every task will assign a id
  TaskKeyMap<int32_t> task_key_id_;

  // id-task map
  std::unordered_map<int32_t, TaskKey> id_task_key_;

  // the taskkey will assign to different group
  std::vector<std::shared_ptr<Group>> groups_;

  // task-group map
  TaskKeyMap<std::shared_ptr<Group>> task_key_group_;

  // wait for Group task
  // when get task from queue the task will put into waiting_group_pool_
  // than get task from waiting_group_pool_ to check whether all task from one group has ready.
  // if yes, put the ready task into waiting_request_pool_, waiting_request_pool_ means the task has ready in this rank
  // should wait other rank ready
  TaskKeyMap<Task> waiting_group_pool_;

  // the task in this pool means already pass group check, waiting other rank ready
  TaskKeyMap<Task>  waiting_request_pool_;

  uint8_t *recvbuf = nullptr;
  int32_t recvbuf_size = 0;

private:
  // exchange string between ranks
  std::vector<std::string> exchange_string(MPI_Comm mpi_comm, int rank, int size, std::string &str);

  // format tskcell to string
  std::string dump_tasks_cells_to_json(const std::vector<TaskCell> &task_cells);

  // parse task cell from json
  std::vector<TaskCell> parse_json_to_task_cells(const std::string &json_str);

  // update group and task cell
  void update(const Context &context, const std::vector<Task> &ready_task);

public:
  Communicator();

  ~Communicator();

  // at here will exchange with other rank get ready task
  std::unordered_map<TaskType, std::vector<Task>> exchange(const Context &context, 
                                                           moodycamel::ConcurrentQueue<Task> &task_queue);
};

}

#endif
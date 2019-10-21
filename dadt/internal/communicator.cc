#include <memory>
#include <mpi.h>

#include "json11.hpp"

#include "definition.h"
#include "context.h"
#include "lock_tensor.h"
#include "communicator.h"

namespace dadt {

Communicator::Communicator() {
}

Communicator::~Communicator() {
  if (nullptr != recvbuf) {
    free(recvbuf);
  }

  recvbuf = nullptr;
}

// exchange string with special MPI_Comm
// return is a string array corresponding the rank index
std::vector<std::string> Communicator::exchange_string(MPI_Comm mpi_comm, int rank, int size, std::string &str) {
  // step1 get all string length
  std::vector<int> str_length((size_t)size);
  str_length[rank] = (int)str.size();

  // get string length
  MPI_CALL(MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, str_length.data(), 1, MPI_INT, mpi_comm));

  // step2 get all string from all process
  int total_length = 0;
  for (auto i : str_length) {
    total_length += i;
  }

  char* all_gather_str_c = (char*)malloc(total_length + 1);
  int *recvcounts = (int*)malloc(sizeof(int) * size);
  int *displs = (int*)malloc(sizeof(int) * size);

  for (int i = 0; i < size; ++i) {
    recvcounts[i] = str_length[i];

    if (0 == i) {
      displs[i] = 0;
    } else {
      displs[i] = displs[i - 1] + recvcounts[i - 1];
    }
  }

  // step3, get all string
  MPI_CALL(MPI_Allgatherv(str.data(), (int)str.size(), MPI_CHAR, all_gather_str_c, recvcounts, displs, MPI_CHAR, mpi_comm));

  all_gather_str_c[total_length] = '\0';

  std::string all_gather_str(all_gather_str_c);

  // free memory
  free(displs);
  free(recvcounts);
  free(all_gather_str_c);

  std::vector<std::string> ret;

  int offset = 0;

  for (int i = 0; i < size; ++i) {
    ret.emplace_back(all_gather_str.substr(offset, str_length[i]));

    offset += str_length[i];
  }

  return std::move(ret);
}

// format tskcell to string
std::string Communicator::dump_tasks_cells_to_json(const std::vector<TaskCell> &task_cells) {
  // format:
  // [
  //  {"task_type":0, "name":b, "num_bytes":20 },
  //  {"task_type":1, "name":a, "num_bytes":20 },
  // ]
  json11::Json::array values;

  for (auto &cell : task_cells) {
    json11::Json::object obj;

    obj["task_type"] = json11::Json(cell.task_type);
    obj["name"]      = json11::Json(cell.name);
    obj["num_bytes"] = json11::Json(cell.num_bytes);

    values.emplace_back(obj);
  }

  json11::Json jsonValue(values);

  return std::move(jsonValue.dump());
}

// parse task from json
std::vector<TaskCell> Communicator::parse_json_to_task_cells(const std::string &json_str) {
  std::vector<TaskCell> task_cells;

  std::string err;
  auto json_object = json11::Json::parse(json_str, err);

  ARGUMENT_CHECK(err.empty(), "parse json get error:" << err);
  ARGUMENT_CHECK(json_object.is_array(), "parse json get error");

  for (auto &cell_obj : json_object.array_items()) {
    ARGUMENT_CHECK(cell_obj.is_object(), "parge json get error");

    auto items = cell_obj.object_items();

    if (items.find("task_type") == items.end() || items.find("name") == items.end() || items.find("num_bytes") == items.end()) {
      RUNTIME_ERROR("parse json get error");
    }

    TaskCell cell;
    cell.task_type = items["task_type"].int_value();
    cell.name      = items["name"].string_value();
    cell.num_bytes = items["num_bytes"].int_value();

    task_cells.emplace_back(std::move(cell));
  }

  return std::move(task_cells);
}

// update group and task cell
void Communicator::update(const Context &context, const std::vector<Task> &ready_tasks) {
  // before update the task info, will remove unused broadcast task, because broadcast only used once at beginning.
  // so avoid exhange to many message between ranks.
  std::vector<TaskCell> new_task_cells;

  for (auto &cell : task_cells_) {
    auto key = std::make_tuple(cell.task_type, cell.name);

    // do not keep kBroadCastTaskType
    if (kBroadCastTaskType == cell.task_type 
        && waiting_group_pool_.find(key) == waiting_group_pool_.end()
        && waiting_request_pool_.find(key) == waiting_request_pool_.end()) {
      continue;
    }

    new_task_cells.emplace_back(cell);
  }

  // insert ready task
  // new_task_cells maybe has duplicate cell
  for (auto &task : ready_tasks) {
    TaskCell cell;

    cell.task_type = task.task_type;
    cell.name = task.name;

    if (nullptr != task.tensor) {
      cell.num_bytes = task.tensor->num_bytes();
    } else {
      cell.num_bytes = 0;
    }

    new_task_cells.emplace_back(std::move(cell));
  }

  // clean old data
  task_cells_.clear();
  task_key_id_.clear();
  id_task_key_.clear();
  groups_.clear();
  task_key_group_.clear();

  // exchange task cell between ranks
  auto task_cell_json = dump_tasks_cells_to_json(new_task_cells);
  auto task_cell_jsons = exchange_string(context.world_comm, context.world_rank, context.world_size, task_cell_json);

  for (auto &json : task_cell_jsons) {
    if (json.empty()) {
      continue;
    }

    auto cells = parse_json_to_task_cells(json);

    for (auto &cell : cells) {
      auto key = std::make_tuple(cell.task_type, cell.name);

      // maybe include duplicate key
      if (task_key_id_.find(key) == task_key_id_.end()) {
        int32_t id = task_cells_.size();

        task_key_id_[key] = id;
        id_task_key_[id] = key;

        task_cells_.emplace_back(cell);
      }
    }
  }

  // for now all rank has the same task cell
  // will group the taskcell by num_bytes
  std::unordered_map<TaskType, std::vector<TaskCell>> category_cells;

  for (auto &cell : task_cells_) {
    category_cells[cell.task_type].emplace_back(cell);
  }

  // the task cell in category_cells is sorted by time (time:get out of queue)
  for (auto &category : category_cells) {
    if (kShutDownTaskType == category.first || kBroadCastTaskType == category.first) {
      // kShutDownTaskType and kBroadCastTaskType group only include one taskkey
      for (auto &cell : category.second) {
        auto key = std::make_tuple(cell.task_type, cell.name);

        auto group = std::make_shared<Group>();

        group->insert_to_aggregate(key);
        groups_.emplace_back(group);

        task_key_group_[key] = group;
      }
    } else if (kAllReduceTaskType == category.first) {
      // all reduce will group by buffer
      for (size_t i = 0; i < category.second.size(); ) {
        if (category.second[i].num_bytes >= context.group_buffer_size) {
          auto key = std::make_tuple(category.second[i].task_type, category.second[i].name);

          auto group = std::make_shared<Group>();

          group->insert_to_aggregate(key);
          groups_.emplace_back(group);

          task_key_group_[key] = group;

          ++i;

          continue;
        }

        auto group = std::make_shared<Group>();

        size_t current_buffer_size = 0;
        size_t j = i;

        for (; j < category.second.size(); ++j) {
          if (current_buffer_size + category.second[j].num_bytes > context.group_buffer_size) {
            break;
          }

          auto key = std::make_tuple(category.second[j].task_type, category.second[j].name);

          group->insert_to_aggregate(key);

          task_key_group_[key] = group;

          current_buffer_size += category.second[j].num_bytes;
        }

        groups_.emplace_back(group);

        i = j;
      }
    } else {
      RUNTIME_ERROR("task type is not support" << category.first);
    }
  }

  // after update, the Group has been empty, so should put waiting_group_pool_ in group again
  // because the group has beeen changed, so maybe this time will get waiting_request_task
  std::vector<TaskKey> should_request_tasks;

  for (auto &item : waiting_group_pool_) {
    ARGUMENT_CHECK(task_key_group_.find(item.first) != task_key_group_.end(), "can not find task key in task_key_group_");

    auto grouped_task_keys = task_key_group_[item.first]->insert_to_ready(item.first);

    for (auto &key : grouped_task_keys) {
      should_request_tasks.emplace_back(key);
    }
  }

  // for now should get task from waiting_group_pool_ put into waiting_request_pool_ by should_request_tasks
  for (auto &key : should_request_tasks) {
    ARGUMENT_CHECK(waiting_group_pool_.find(key) != waiting_group_pool_.end(), "waiting group do not have taskkey");
    ARGUMENT_CHECK(waiting_request_pool_.find(key) == waiting_request_pool_.end(), "waiting_request_pool_ has already has the task");

    // put into waiting_request_pool_
    waiting_request_pool_[key] = waiting_group_pool_[key];
    waiting_group_pool_.erase(key);
  }
}

// at here will exchange with other rank get ready task
std::unordered_map<TaskType, std::vector<Task>> Communicator::exchange(const Context &context, 
                                                                       moodycamel::ConcurrentQueue<Task> &task_queue) {
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

  // after get task from queue, will check whether the taskkey in the map, if not will communicator with other rank to exchange new taskkey
  uint8_t unknow_task_key = 0;

  for (auto &task : ready_tasks) {
    auto key = std::make_tuple(task.task_type, task.name);

    if (task_key_id_.find(key) == task_key_id_.end()) {
      unknow_task_key = 1;
      break;
    }
  }

  // use "allreduce or" to check whether have unknow taskkey
  MPI_CALL(MPI_Allreduce(MPI_IN_PLACE, &unknow_task_key, 1, MPI_UINT8_T, MPI_LOR, context.world_comm));

  if (0 != unknow_task_key) {
    // for now we get some unknow task key, than we will exchange the taskkey between ranks and update info.
    update(context, ready_tasks);
  }

  // for now the info has been updated.
  // firstly put ready_task into waiting_group_pool_
  for (auto &task : ready_tasks) {
    auto key = std::make_tuple(task.task_type, task.name);

    ARGUMENT_CHECK(waiting_group_pool_.find(key) == waiting_group_pool_.end(), "waiting_group_pool_ has already include the task:" << task.name);

    waiting_group_pool_[key] = task;
  }

  // check the task will be request with other rank
  std::vector<TaskKey> should_request_tasks;

  for (auto &task : ready_tasks) {
    auto key = std::make_tuple(task.task_type, task.name);

    auto grouped_task_keys = task_key_group_[key]->insert_to_ready(key);

    for (auto &t : grouped_task_keys) {
      should_request_tasks.emplace_back(t);
    }
  }

  for (auto &key : should_request_tasks) {
    ARGUMENT_CHECK(waiting_group_pool_.find(key) != waiting_group_pool_.end(), "waiting group do not have task:" << std::get<1>(key));
    ARGUMENT_CHECK(waiting_request_pool_.find(key) == waiting_request_pool_.end(), "waiting_request_pool_ has already has the task:" << std::get<1>(key));

    // put into waiting_request_pool_
    waiting_request_pool_[key] = waiting_group_pool_[key];
    waiting_group_pool_.erase(key);
  }

  // now all waiting request task has been put into waiting_request_pool_
  // should talk with other rank to get real ready tasks
  int total_task_count = task_cells_.size();
  int byte_count = (total_task_count + 7) / 8;

  if (byte_count > recvbuf_size) {
    if (nullptr != recvbuf) {
      free(recvbuf);
    }

    recvbuf_size = byte_count;
    recvbuf = (uint8_t*)malloc(sizeof(uint8_t) * recvbuf_size);
  }

  memset(recvbuf, 0, byte_count);

  // iterator waiting_request_pool_ set corresponding bit to be 1
  for (auto &task : waiting_request_pool_) {
    ARGUMENT_CHECK(task_key_id_.find(task.first) != task_key_id_.end(), "can not find task id");

    auto id = task_key_id_[task.first];

    ARGUMENT_CHECK(0 <= id && id < total_task_count, "task id out of range");

    // set bit to be 1
    int byte_index = id / 8;
    int bit_offset = id % 8;

    recvbuf[byte_index] |= (((uint8_t)1) << bit_offset);
  }

  // use allreduce to check whether other rank ready
  MPI_CALL(MPI_Allreduce(MPI_IN_PLACE, recvbuf, byte_count, MPI_UINT8_T, MPI_BAND, context.world_comm));

  std::unordered_map<TaskType, std::vector<Task>> should_execute_tasks;

  // iterator the recvbuf to check whether task should be execute
  for (int id = 0; id < total_task_count; ++id) {
    int byte_index = id / 8;
    int bit_offset = id % 8;

    if (0 == (recvbuf[byte_index] & (((uint8_t)1) << bit_offset))) {
      continue;
    }

    ARGUMENT_CHECK(id_task_key_.find(id) != id_task_key_.end(), "can not find id:" << id);

    // get task key
    auto task_key = id_task_key_[id];

    ARGUMENT_CHECK(waiting_request_pool_.find(task_key) != waiting_request_pool_.end(), "cann not find task:" << std::get<1>(task_key) << " in waiting_request_pool_.");

    auto task = waiting_request_pool_[task_key];

    should_execute_tasks[task.task_type].emplace_back(std::move(task));

    waiting_request_pool_.erase(task_key);
  }

  // the task in should_execute_tasks is sorted by id
  return std::move(should_execute_tasks);
}

}
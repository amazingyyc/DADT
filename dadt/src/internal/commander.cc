#include <iostream>
#include <vector>
#include <string>
#include <stdexcept>

#include <mpi.h>

#include "json11.hpp"

#include "exception.h"
#include "commander.h"
#include "task.h"
#include "lock_tensor.h"

namespace dadt {

Commander::Commander() : initialized_(false), worker_stopped_(false) {}

// initialize context
// the funtion should call in background thread
void Commander::initialize_context() {
  int thread_required = MPI_THREAD_MULTIPLE;
  int thread_provided;

  // init mpi
  MPI_CALL(MPI_Init_thread(NULL, NULL, thread_required, &thread_provided));

  // dump a world comm
  MPI_CALL(MPI_Comm_dup(MPI_COMM_WORLD, &context_.world_comm));

  // get rank and size
  MPI_CALL(MPI_Comm_rank(context_.world_comm, &context_.world_rank));
  MPI_CALL(MPI_Comm_size(context_.world_comm, &context_.world_size));

  // is a leader
  context_.is_leader = (0 == context_.world_rank);

  // init local comm
  MPI_CALL(MPI_Comm_split_type(context_.world_comm, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &context_.local_comm));

  // get local rank and local size
  MPI_CALL(MPI_Comm_rank(context_.local_comm, &context_.local_rank));
  MPI_CALL(MPI_Comm_size(context_.local_comm, &context_.local_size));

  // local leader
  context_.is_local_leader = (0 == context_.local_rank);

  // init cross comm
  MPI_CALL(MPI_Comm_split(context_.world_comm, context_.local_rank, context_.world_rank, &context_.cross_comm));
  MPI_CALL(MPI_Comm_rank(context_.cross_comm, &context_.cross_rank));
  MPI_CALL(MPI_Comm_size(context_.cross_comm, &context_.cross_size));

  context_.is_cross_leader = (0 == context_.cross_rank);

  // read environment variable DADT_ALLREDUCE_AVERAGE_DISABLE
  auto allreduce_average_disable = std::getenv(DADT_ALLREDUCE_AVERAGE_DISABLE);
  if (nullptr != allreduce_average_disable) {
    context_.allreduce_average_disbale = true;
  }

  // read cycle time
  auto cycle_duration_ms = std::getenv(DADT_CYCLE_DURATION_MS);
  if (nullptr != cycle_duration_ms) {
    context_.cycle_duration_millisecond = std::strtoll(cycle_duration_ms, nullptr, 10);
    context_.cycle_duration_microsecond = context_.cycle_duration_millisecond * 1000;
  } else {
    context_.cycle_duration_millisecond = 5;
    context_.cycle_duration_microsecond = context_.cycle_duration_millisecond * 1000;
  }

  // set the flag
  initialized_ = true;
}

// exchange string with special MPI_Comm
// return is a string array corresponding the rank index
std::vector<std::string> Commander::exchange_string(MPI_Comm mpi_comm, int rank, int size, std::string &str) {
  // step1 get all string lenght
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

// formate a category task to json str
std::string Commander::dump_tasks_to_json(const std::unordered_map<TaskType, std::vector<std::string>> &category_tasks) {
  json11::Json::object document;

  for (auto &item : category_tasks) {
    json11::Json::array value;

    for (auto &name : item.second) {
      value.emplace_back(json11::Json(name));
    }

    document[std::to_string(item.first)] = json11::Json(value);
  }

  std::string json_str;
  json11::Json(document).dump(json_str);

  return std::move(json_str);
}

// parse task from json
std::unordered_map<TaskType, std::vector<std::string>> Commander::parse_json_to_tasks(const std::string &json_str) {
  std::unordered_map<TaskType, std::vector<std::string>> category_tasks;

  std::string err;
  auto json_object = json11::Json::parse(json_str, err);

  ARGUMENT_CHECK(err.empty(), "parse json get error:" << err);
  ARGUMENT_CHECK(json_object.is_object(), "parse json get error");

  for (auto &json_item : json_object.object_items()) {
    TaskType task_type;

    try {
      task_type = std::stoi(json_item.first);
    } catch (std::invalid_argument&) {
      RUNTIME_ERROR("parse string to int get invalid_argument");
    } catch (std::out_of_range&) {
      RUNTIME_ERROR("parse string to int get out_of_range");
    }

    ARGUMENT_CHECK(json_item.second.is_array(), "parse json get error");

    for (auto &id_json : json_item.second.array_items()) {
      ARGUMENT_CHECK(id_json.is_string(), "parge json get error");
      auto name = id_json.string_value();

      category_tasks[task_type].emplace_back(name);
    }
  }

  return std::move(category_tasks);
}

// insert a ready tensor and decide if it it ready to execute
// only if all process has ready to do the task
bool Commander::if_execute_task(int rank, TaskType task_type, std::string name) {
  std::tuple<TaskType, std::string> key = std::make_tuple(task_type, name);

  task_register_[key].insert(rank);

  if (context_.world_size == task_register_[key].size()) {
    task_register_[key].clear();
    return true;
  }

  return false;
}

// exchange the tasks with each process
std::unordered_map<TaskType, std::vector<Task>> Commander::exchange_execute_tasks(std::vector<Task> &tasks) {
  // step1: put all task in task_pool
  for (auto &t : tasks) {
    auto key = std::make_tuple(t.task_type, t.name);
    task_pool_[key] = t;
  }

  //step2: exchange the tasks
  // the input will package to be a json-fromat string to send with each other
  // json format
  // key: the TaskType string
  // value: tensor ids
  // {
  //   "0" : ["0", "1", "2"],
  //   "2" : ["3", "4", "5"],
  // }
  std::unordered_map<TaskType, std::vector<std::string>> category_tasks;

  for (auto &t : tasks) {
    category_tasks[t.task_type].emplace_back(t.name);
  }

  // formate to s string
  std::string tasks_json = dump_tasks_to_json(category_tasks);

  // step3: exchange with other process
  auto all_tasks_json = exchange_string(context_.world_comm, context_.world_rank, context_.world_size, tasks_json);

  ARGUMENT_CHECK(all_tasks_json.size() == context_.world_size, "exchange_string get error");

  // step4: parse json str and get the executed task
  // for now every process have get all the tasks
  // than iterator to get all task and decide which task will be executed
  std::unordered_map<TaskType, std::vector<Task>> will_execute_tasks;

  for (int rank = 0; rank < all_tasks_json.size(); ++rank) {
    auto execute_tasks = parse_json_to_tasks(all_tasks_json[rank]);

    for (auto &item : execute_tasks) {
      auto task_type = item.first;

      for (auto &name : item.second) {
        if (if_execute_task(rank, task_type, name)) {
          auto key = std::make_tuple(task_type, name);

          ARGUMENT_CHECK(task_pool_.find(key) != task_pool_.end(), "the task in not in task_pool");

          will_execute_tasks[task_type].emplace_back(task_pool_[key]);

          // remove it from pool
          task_pool_.erase(key);
        }
      }
    }
  }

  // for now we have get all task that will be executed at this time
  // every process have the same ready [task_type, tensor_ids]  
  return std::move(will_execute_tasks);
}

// init the commander
void Commander::initialize() {
  ARGUMENT_CHECK(false == initialized_, "can not initialize twice");

  // init a thread and wait finish init context
  worker_thread_ = std::thread(&Commander::worker_do_cycle, this);

  while (false == initialized_) {/**just wait*/
  }
}

// if the commander have been initialized
bool Commander::initialized() {
  return initialized_.load();
}

// process count
int Commander::size() {
  ARGUMENT_CHECK(initialized(), "the commander has not initialized");

  return context_.world_size;
}

// local machine process count
int Commander::local_size() {
  ARGUMENT_CHECK(initialized(), "the commander has not initialized");

  return context_.local_size;
}

// process rank
int Commander::rank() {
  ARGUMENT_CHECK(initialized(), "the commander has not initialized");

  return context_.world_rank;
}

// local rank
int Commander::local_rank() {
  ARGUMENT_CHECK(initialized(), "the commander has not initialized");

  return context_.local_rank;
}

// barrier all process
void Commander::barrier() {
  ARGUMENT_CHECK(initialized(), "the commander has not initialized");

  MPI_CALL(MPI_Barrier(context_.world_comm));
}

// local barrier process
void Commander::local_barrier() {
  ARGUMENT_CHECK(initialized(), "the commander has not initialized");

  MPI_CALL(MPI_Barrier(context_.local_comm));
}

// insert a message to message_queue_
void Commander::enqueue_task(Task &&t) {
  auto ret = task_queue_.enqueue(t);

  ARGUMENT_CHECK(ret, "enqueue task to queue get error!");
}

// get the message from the queue and allreduce cross all node
void Commander::worker_do_task() {
  std::vector<Task> tasks;

  // step 1: get all task from the queue
  while (true) {
    Task t;
    if (task_queue_.try_dequeue(t)) {
      tasks.emplace_back(t);
    } else {
      break;
    }
  }

  // step2, exchange with other node
  auto execute_tasks = exchange_execute_tasks(tasks);

  // for now every process have some task that will be executed
  // step3, sort the the tasks by TaskType
  std::vector<TaskType> execute_task_types;
  for (auto &item : execute_tasks) {
    execute_task_types.emplace_back(item.first);
  }

  // sort the TaskType
  std::sort(execute_task_types.begin(), execute_task_types.end(), [](const TaskType &t1, const TaskType &t2) {
    return t1 > t2;
  });

  for (auto &task_type : execute_task_types) {
    // sort the tensor by id and all process have the same order
    auto &tasks = execute_tasks[task_type];

    // sort by name
    std::sort(tasks.begin(), tasks.end(), [](const Task &t1, const Task &t2) {
      return t1.name > t2.name;
    });

    // use the corresponding executor to do the task
    (*task_executors_[task_type])(context_, tasks);

    // after execute the task callback
    for (auto &t : tasks) {
      t.done();
    }
  }
}

// used for background_thread_ to do the task
void Commander::worker_do_cycle() {
  // init context
  initialize_context();

  while (false == worker_stopped_) {
    auto task_start_time = std::chrono::steady_clock::now();

    worker_do_task();

    auto task_duration = std::chrono::steady_clock::now() - task_start_time;

    if (task_duration < std::chrono::microseconds(context_.cycle_duration_microsecond)) {
      std::this_thread::sleep_for(std::chrono::microseconds(context_.cycle_duration_microsecond) - task_duration);
    }
  }
}

void Commander::stop_worker() {
  worker_stopped_ = true;

  // wait worker stop
  worker_thread_.join();
}

// get a interim tensor by TaskType
std::shared_ptr<LockTensor> Commander::get_interim_tensor(TaskType task_type,
                                                          std::string name,
                                                          std::vector<int> dims,
                                                          ElementType element_type) {
  ARGUMENT_CHECK(initialized(), "the commander has not initialized");

  return task_executors_[task_type]->get_interim_tensor(name, dims, element_type);
}

}
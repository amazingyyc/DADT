#include <iostream>
#include <vector>
#include <string>
#include <stdexcept>

#include <mpi.h>

#ifdef HAVE_NCCL
#include <cuda_runtime.h>
#include <nccl.h>
#endif

#include "json11.hpp"

#include "definition.h"
#include "commander.h"
#include "task.h"
#include "task_executor.h"
#include "lock_tensor.h"
#include "mpi_all_reduce_executor.h"
#include "mpi_broad_cast_executor.h"

#ifdef HAVE_NCCL 
#include "nccl_broad_cast_executor.h"
#include "nccl_all_reduce_executor.h"
#include "mpi_cuda_broad_cast_executor.h"
#include "mpi_cuda_all_reduce_executor.h"
#endif

namespace dadt {

Commander::Commander() : initialized_(false) {}

// initialize context
// the funtion should call in background thread
void Commander::init_context(Config config) {
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

  // create float16 data type
  MPI_Datatype MPI_FLOAT16_T;
  MPI_Type_contiguous(2, MPI_BYTE, &MPI_FLOAT16_T);
  MPI_Type_commit(&MPI_FLOAT16_T);

  context_.MPI_FLOAT16_T = MPI_FLOAT16_T;

#ifdef HAVE_NCCL
  context_.gpu_device_id = context_.local_rank;

  // set CPU device
  CUDA_CALL(cudaSetDevice(context_.gpu_device_id));

  // cuda stream
  int priority;

  CUDA_CALL(cudaDeviceGetStreamPriorityRange(nullptr, &priority));
  CUDA_CALL(cudaStreamCreateWithPriority(&(context_.cuda_stream), cudaStreamNonBlocking, priority));

  // after create the mpi comm
  // will create nccl context
  ncclUniqueId nccl_id;
  if (0 == context_.world_rank) {
    NCCL_CALL(ncclGetUniqueId(&nccl_id));
  }

  // broad cast to other rank
  MPI_CALL(MPI_Bcast((void*)&nccl_id, sizeof(nccl_id), MPI_BYTE, 0, context_.world_comm));

  ncclComm_t nccl_comm;
  // init nccl comm
  NCCL_CALL(ncclCommInitRank(&nccl_comm, context_.world_size, nccl_id, context_.world_rank));

  context_.nccl_id   = nccl_id;
  context_.nccl_comm = nccl_comm;

  MPI_CALL(MPI_Barrier(context_.world_comm));
#endif

  // get cpu device and gpu device
  auto cpu_device = get_cpu_device();

#ifdef HAVE_NCCL
  auto gpu_device = get_gpu_device(context_.gpu_device_id);
#endif

  // set cycle time
  if (config.cycle_duration_ms <= 0) {
    config.cycle_duration_ms = 5;
  }

  // convert to microsecond
  context_.cycle_duration_us = config.cycle_duration_ms * 1000;

  // create broadcast executor
  if (0 == config.broad_cast_executor) {
    task_executors_[kBroadCastTaskType] = std::make_shared<MPIBroadCastExecutor>(cpu_device);
#ifdef HAVE_NCCL
  } else if (1 == config.broad_cast_executor) {
    task_executors_[kBroadCastTaskType] = std::make_shared<NCCLBroadCastExecutor>(gpu_device);
  } else if (2 == config.broad_cast_executor) {
    task_executors_[kBroadCastTaskType] = std::make_shared<MPICUDABroadCastExecutor>(gpu_device);
#endif
  } else {
    RUNTIME_ERROR("broad_cast_executor is:" << config.broad_cast_executor << " not support, please set to be mpi, nccl (for GPU), mpicuda (for GPU).");
  }

  if (0 == config.all_reduce_executor) {
    task_executors_[kAllReduceTaskType] = std::make_shared<MPIAllReduceExecutor>(cpu_device, config.all_reduce_buffer_size);
#ifdef HAVE_NCCL
  } else if (1 == config.all_reduce_executor) {
    task_executors_[kAllReduceTaskType] = std::make_shared<NCCLAllReduceExecutor>(gpu_device, config.all_reduce_buffer_size);
  } else if (2 == config.all_reduce_executor) {
    task_executors_[kAllReduceTaskType] = std::make_shared<MPICUDAAllReduceExecutor>(gpu_device, config.all_reduce_buffer_size);
#endif
  } else {
    RUNTIME_ERROR("all_reduce_executor is:" << config.all_reduce_executor << " not support, please set to be mpi, nccl (for GPU), mpicuda (for GPU).");
  }

  // whether enable timeline
  // only rank 0 will write timeline
  if (0 == context_.world_rank && nullptr != config.timeline_path) {
    // enable timeline
    context_.enable_timeline = true;

    // file path
    std::string path(config.timeline_path);

    // create a timeline
    timeline_ = std::make_shared<TimeLine>(path);
  } else {
    context_.enable_timeline = false;
  }

  // set the flag
  initialized_ = true;
}

void Commander::clear_context() {
  // clean executor, every executor should clean it's resource when call deinit function
  task_executors_.clear();

#ifdef HAVE_NCCL
  // clean cuda event
  for (auto &item : op_cuda_events_) {
    CUDA_CALL(cudaEventDestroy(item.second));
  }

  op_cuda_events_.clear();

  // clear cuda and nccl resource
  NCCL_CALL(ncclCommDestroy(context_.nccl_comm));
  CUDA_CALL(cudaStreamDestroy(context_.cuda_stream));
#endif

  // clean mpi
  MPI_CALL(MPI_Type_free(&context_.MPI_FLOAT16_T));
  MPI_CALL(MPI_Comm_free(&context_.cross_comm));
  MPI_CALL(MPI_Comm_free(&context_.local_comm));
  MPI_CALL(MPI_Comm_free(&context_.world_comm));
  MPI_CALL(MPI_Finalize());
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
bool Commander::check_execute_task(int rank, TaskType task_type, std::string name) {
  std::tuple<TaskType, std::string> key = std::make_tuple(task_type, name);

  task_register_[key].insert(rank);

  if (context_.world_size == task_register_[key].size()) {
    task_register_.erase(key);
    return true;
  }

  return false;
}

// exchange the tasks with each process
std::unordered_map<TaskType, std::vector<Task>> Commander::exchange_execute_tasks(std::vector<Task> &tasks) {
  // timeline kStayInTaskPoolEvent
  if (context_.enable_timeline.load()) {
    timeline_->begin(tasks, kStayInTaskPoolEvent);
  }

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

  // format to string
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
        if (check_execute_task(rank, task_type, name)) {
          auto key = std::make_tuple(task_type, name);

          ARGUMENT_CHECK(task_pool_.find(key) != task_pool_.end(), "the task not in task_pool");

          will_execute_tasks[task_type].emplace_back(task_pool_[key]);

          // remove it from pool
          task_pool_.erase(key);
        }
      }
    }
  }

  // timeline
  if (context_.enable_timeline.load()) {
    for (auto &item : will_execute_tasks) {
      timeline_->end(item.second, kStayInTaskPoolEvent);
    }
  }

  // for now we have get all task that will be executed at this time
  // every process have the same ready [task_type, tensor_ids]  
  return std::move(will_execute_tasks);
}

// init the commander
void Commander::init(Config config) {
  ARGUMENT_CHECK(false == initialized_, "can not initialize twice");

  // init a thread and wait finish init context
  worker_thread_ = std::thread(&Commander::worker_do_cycle, this, config);

  while (false == initialized_) {/**just wait*/}
}

// shutdown background thread
void Commander::shutdown() {
  ARGUMENT_CHECK(initialized_, "the commander has not initialized");

  // stop worker thread
  // put a shutdown task in queue
  Task shutdown_task;
  shutdown_task.task_type = kShutDownTaskType;
  shutdown_task.name      = DADT_SHUTDOWN_TASK_NAME;

  // shutdown system
  enqueue_task(std::move(shutdown_task));

  worker_thread_.join();

  // set flag
  initialized_ = false;
}

// whether the commander have been initialized
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
  ARGUMENT_CHECK(initialized(), "the commander has not initialized");

    // timeline
  if (context_.enable_timeline.load()) {
    timeline_->begin(t.name, kStayInTaskQueueEvent);
  }

  auto ret = task_queue_.enqueue(t);

  ARGUMENT_CHECK(ret, "enqueue task to queue get error!");
}

// timeline event
void Commander::begin_timeline_event(const std::string &name, const std::string &event) {
  if (context_.enable_timeline.load()) {
    timeline_->begin(name, event);
  }
}

void Commander::end_timeline_event(const std::string &name, const std::string &event) {
  if (context_.enable_timeline.load()) {
    timeline_->end(name, event);
  }
}

#ifdef HAVE_NCCL
cudaEvent_t Commander::obtain_cuda_event() {
  std::unique_lock<std::mutex> lock(op_cuda_event_mutex_);

  auto id = std::this_thread::get_id();

  if (op_cuda_events_.find(id) == op_cuda_events_.end()) {
    cudaEvent_t event;
    CUDA_CALL(cudaEventCreate(&event));

    op_cuda_events_[id] = event;
  }

  return op_cuda_events_[id];
}
#endif

std::shared_ptr<LockTensor> Commander::obtain_midway_tensor(TaskType task_type, std::string name) {
  ARGUMENT_CHECK(initialized(), "the commander has not initialized");

  return task_executors_[task_type]->obtain_midway_tensor(name);
}

// get a interim tensor by TaskType
std::shared_ptr<LockTensor> Commander::create_midway_tensor(TaskType task_type,
                                                            std::string name,
                                                            std::vector<int> dims,
                                                            ElementType element_type) {
  ARGUMENT_CHECK(initialized(), "the commander has not initialized");

  return task_executors_[task_type]->create_midway_tensor(name, dims, element_type);
}

// get the message from the queue and allreduce cross all node
bool Commander::worker_do_task() {
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

  // timeline out of queue
  if (context_.enable_timeline.load()) {
    timeline_->end(tasks, kStayInTaskQueueEvent);
  }

  // step2, exchange with other node
  auto execute_tasks = exchange_execute_tasks(tasks);

  // check whether shutdown
  bool shutdown = false;

  if (execute_tasks.find(kShutDownTaskType) != execute_tasks.end()) {
    shutdown = true;
    execute_tasks.erase(kShutDownTaskType);
  }

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
    (*task_executors_[task_type])(context_, tasks, timeline_);
  }

  return shutdown;
}

// used for background_thread_ to do the task
void Commander::worker_do_cycle(Config config) {
  // init context
  init_context(config);

  while (true) {
    auto task_start_time = std::chrono::steady_clock::now();

    // do the task
    auto shutdown = worker_do_task();

    if (shutdown) {
      break;
    }

    auto task_duration = std::chrono::steady_clock::now() - task_start_time;

    if (task_duration < std::chrono::microseconds(context_.cycle_duration_us)) {
      std::this_thread::sleep_for(std::chrono::microseconds(context_.cycle_duration_us) - task_duration);
    }
  }

  // when shutdown clear the resource.
  clear_context();
}


}
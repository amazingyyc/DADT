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
  if (config.cycle_duration_ms < 0) {
    config.cycle_duration_ms = 5;
  }

  // convert to microsecond
  context_.cycle_duration_us      = config.cycle_duration_ms * 1000;
  context_.all_reduce_buffer_size = config.all_reduce_buffer_size;
  context_.group_buffer_size      = config.group_buffer_size;

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
    task_executors_[kAllReduceTaskType] = std::make_shared<MPIAllReduceExecutor>(cpu_device, context_.all_reduce_buffer_size);
#ifdef HAVE_NCCL
  } else if (1 == config.all_reduce_executor) {
    task_executors_[kAllReduceTaskType] = std::make_shared<NCCLAllReduceExecutor>(gpu_device, context_.all_reduce_buffer_size);
  } else if (2 == config.all_reduce_executor) {
    task_executors_[kAllReduceTaskType] = std::make_shared<MPICUDAAllReduceExecutor>(gpu_device, context_.all_reduce_buffer_size);
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
  // if (context_.enable_timeline.load()) {
  //   timeline_->begin(t.name, kStayInTaskQueueEvent);
  // }

  auto ret = task_queue_.enqueue(t);

  ARGUMENT_CHECK(ret, "enqueue task to queue get error!");
}

// timeline event
void Commander::begin_timeline_event(const std::string &name, const std::string &event) {
  // if (context_.enable_timeline.load()) {
  //   timeline_->begin(name, event);
  // }
}

void Commander::end_timeline_event(const std::string &name, const std::string &event) {
  // if (context_.enable_timeline.load()) {
  //   timeline_->end(name, event);
  // }
}

#ifdef HAVE_NCCL
cudaEvent_t Commander::obtain_cuda_event() {
  SpinLockHandler handler(op_cuda_events_locker_);

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
  // get should execute task
  auto execute_tasks = communicator_.exchange(context_, task_queue_, timeline_);

  // check whether shutdown
  bool shutdown = false;

  if (execute_tasks.find(kShutDownTaskType) != execute_tasks.end()) {
    shutdown = true;
    execute_tasks.erase(kShutDownTaskType);
  }

  // for now every process have some task that will be executed
  // sort the the tasks by TaskType
  std::vector<TaskType> execute_task_types;
  for (auto &item : execute_tasks) {
    execute_task_types.emplace_back(item.first);
  }

  // sort the TaskType
  std::sort(execute_task_types.begin(), execute_task_types.end(), [](const TaskType &t1, const TaskType &t2) {
    return t1 > t2;
  });

  for (auto &task_type : execute_task_types) {
    // use the corresponding executor to do the task
    (*task_executors_[task_type])(context_, execute_tasks[task_type], timeline_);
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
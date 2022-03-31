#include "commander.h"

#include <mpi.h>

#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef HAVE_NCCL
#include <cuda_runtime.h>
#include <nccl.h>
#endif

#include "common/exception.h"
#include "common/log.h"
#include "common/task.h"
#include "executor/mpi_all_reduce_executor.h"
#include "executor/mpi_broad_cast_executor.h"
#include "executor/mpi_coo_all_reduce_executor.h"
#include "executor/task_executor.h"
#include "t/lock_tensor.h"

#ifdef HAVE_NCCL
#include "executor/nccl_all_reduce_executor.h"
#include "executor/nccl_broad_cast_executor.h"
#include "executor/nccl_coo_all_reduce_executor.h"
#endif

namespace dadt {

Commander::Commander() : initialized_(false) {
}

// initialize context
// the funtion should call in background thread
void Commander::Setup(const Config& config) {
  LOG_INFO("DADT initialize, cycle_duration_ms:["
           << config.cycle_duration_ms << "], executor_type:["
           << config.executor_type << "], all_reduce_buffer_size:["
           << config.all_reduce_buffer_size << "], gpu_device_id:["
           << config.gpu_device_id << "]");

  ARGUMENT_CHECK(config.cycle_duration_ms >= 0,
                 "config.cycle_duration_ms must >= 0");

  // convert to microsecond
  context_.cycle_duration_us = config.cycle_duration_ms * 1000;
  context_.all_reduce_buffer_size = config.all_reduce_buffer_size;
  context_.group_buffer_size = config.group_buffer_size;

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
  MPI_CALL(MPI_Comm_split_type(context_.world_comm, MPI_COMM_TYPE_SHARED, 0,
                               MPI_INFO_NULL, &context_.local_comm));

  // get local rank and local size
  MPI_CALL(MPI_Comm_rank(context_.local_comm, &context_.local_rank));
  MPI_CALL(MPI_Comm_size(context_.local_comm, &context_.local_size));

  // local leader
  context_.is_local_leader = (0 == context_.local_rank);

  // init cross comm
  MPI_CALL(MPI_Comm_split(context_.world_comm, context_.local_rank,
                          context_.world_rank, &context_.cross_comm));
  MPI_CALL(MPI_Comm_rank(context_.cross_comm, &context_.cross_rank));
  MPI_CALL(MPI_Comm_size(context_.cross_comm, &context_.cross_size));

  context_.is_cross_leader = (0 == context_.cross_rank);

  // create float16 data type
  MPI_Datatype MPI_FLOAT16_T;
  MPI_Type_contiguous(2, MPI_BYTE, &MPI_FLOAT16_T);
  MPI_Type_commit(&MPI_FLOAT16_T);

  context_.MPI_FLOAT16_T = MPI_FLOAT16_T;

  if (config.executor_type == "nccl") {
#ifdef HAVE_NCCL
    if (config.gpu_device_id < 0) {
      context_.gpu_device_id = context_.local_rank;
    } else {
      context_.gpu_device_id = config.gpu_device_id;
    }

    // set GPU device
    CUDA_CALL(cudaSetDevice(context_.gpu_device_id));

    // cuda stream
    int priority;

    CUDA_CALL(cudaDeviceGetStreamPriorityRange(nullptr, &priority));
    CUDA_CALL(cudaStreamCreateWithPriority(&(context_.cuda_stream),
                                           cudaStreamNonBlocking, priority));

    // after create the mpi comm
    // will create nccl context
    ncclUniqueId nccl_id;
    if (0 == context_.world_rank) {
      NCCL_CALL(ncclGetUniqueId(&nccl_id));
    }

    // broad cast to other rank
    MPI_CALL(MPI_Bcast((void*)&nccl_id, sizeof(nccl_id), MPI_BYTE, 0,
                       context_.world_comm));

    ncclComm_t nccl_comm;
    // init nccl comm
    NCCL_CALL(ncclCommInitRank(&nccl_comm, context_.world_size, nccl_id,
                               context_.world_rank));

    context_.nccl_id = nccl_id;
    context_.nccl_comm = nccl_comm;

    // Get the default GPU device.
    auto gpu_device = Device::GPUDevice(context_.gpu_device_id);

    // Broadcast executor.
    task_executors_.emplace(
        kBroadCastTaskType,
        std::unique_ptr<NCCLBroadCastExecutor>(new NCCLBroadCastExecutor()));

    // AllReduce executor.
    task_executors_.emplace(
        kAllReduceTaskType,
        std::unique_ptr<NCCLAllReduceExecutor>(new NCCLAllReduceExecutor(
            gpu_device, context_.all_reduce_buffer_size)));

    // CooAllReduce executor.
    task_executors_.emplace(kCooAllReduceTaskType,
                            std::unique_ptr<NCCLCooAllReduceExecutor>(
                                new NCCLCooAllReduceExecutor()));
#else
    RUNTIME_ERROR("DADT not build with CUDA, but executor type is 'nccl'");
#endif
  } else if (config.executor_type == "mpi") {
    auto cpu_device = Device::CPUDevice();

    task_executors_.emplace(
        kBroadCastTaskType,
        std::unique_ptr<MPIBroadCastExecutor>(new MPIBroadCastExecutor()));

    task_executors_.emplace(
        kAllReduceTaskType,
        std::unique_ptr<MPIAllReduceExecutor>(new MPIAllReduceExecutor(
            cpu_device, context_.all_reduce_buffer_size)));

    task_executors_.emplace(kCooAllReduceTaskType,
                            std::unique_ptr<MPICooAllReduceExecutor>(
                                new MPICooAllReduceExecutor()));
  } else {
    RUNTIME_ERROR("executor_type is:"
                  << config.executor_type
                  << " not support, please set to be 'mpi', 'nccl'(for GPU).");
  }

  MPI_CALL(MPI_Barrier(context_.world_comm));
}

void Commander::Clear() {
  // clean mpi
  MPI_CALL(MPI_Type_free(&context_.MPI_FLOAT16_T));
  MPI_CALL(MPI_Comm_free(&context_.cross_comm));
  MPI_CALL(MPI_Comm_free(&context_.local_comm));
  MPI_CALL(MPI_Comm_free(&context_.world_comm));
  MPI_CALL(MPI_Finalize());
}

bool Commander::DoTask() {
  std::unordered_map<TaskType, std::vector<Task>> execute_tasks =
      communicator_.Exchange(context_, task_queue_);

  bool shutdown = false;
  if (execute_tasks.find(kShutDownTaskType) != execute_tasks.end()) {
    shutdown = true;
    execute_tasks.erase(kShutDownTaskType);
  }

  // for now every rank have some task that will be executed
  // sort the the tasks by TaskType
  std::vector<TaskType> execute_task_types;
  for (const auto& [task_type, _] : execute_tasks) {
    execute_task_types.emplace_back(task_type);
  }

  // sort the TaskType
  std::sort(execute_task_types.begin(), execute_task_types.end(),
            [](const TaskType& t1, const TaskType& t2) { return t1 > t2; });

  // sort Task to make sure every rank has the same sequence.
  for (auto& [_, tasks] : execute_tasks) {
    std::sort(tasks.begin(), tasks.end(), [](const Task& t1, const Task& t2) {
      if (t1.type == t2.type) {
        return t1.id > t2.id;
      }

      return t1.type > t2.type;
    });
  }

  for (const auto& task_type : execute_task_types) {
    task_executors_[task_type]->Do(context_, execute_tasks[task_type]);
  }

  return shutdown;
}

// used for background_thread_ to do the task
void Commander::Run(const Config& config) {
  // init context
  Setup(config);

  // set the flag
  initialized_.store(true);

  while (true) {
    auto task_start_time = std::chrono::steady_clock::now();

    // do the task
    bool shutdown = DoTask();
    if (shutdown) {
      break;
    }

    auto task_duration = std::chrono::steady_clock::now() - task_start_time;

    if (task_duration < std::chrono::microseconds(context_.cycle_duration_us)) {
      std::this_thread::sleep_for(
          std::chrono::microseconds(context_.cycle_duration_us) -
          task_duration);
    }
  }

  // when shutdown clear the resource.
  Clear();
}

// init The commander
void Commander::Initialize(const Config& config) {
  ARGUMENT_CHECK(initialized_.load() == false, "Can not initialize twice!");

  // init a thread and wait finish init context
  worker_thread_ = std::thread(&Commander::Run, this, config);

  while (initialized_.load() == false) {
  }
}

// shutdown background thread
void Commander::Shutdown() {
  ARGUMENT_CHECK(initialized_.load(), "The commander has not initialized");

  // stop worker thread
  // put a shutdown task in queue
  Task shutdown_task;
  shutdown_task.type = kShutDownTaskType;
  shutdown_task.id = kShutdowId;

  // shutdown system
  EnqueueTask(std::move(shutdown_task));

  if (worker_thread_.joinable()) {
    worker_thread_.join();
  }

  // set flag
  initialized_ = false;
}

// whether The commander have been initialized
bool Commander::Initialized() const {
  return initialized_.load();
}

// process count
int Commander::Size() const {
  ARGUMENT_CHECK(initialized_.load(), "The commander has not initialized");

  return context_.world_size;
}

// local machine process count
int Commander::LocalSize() const {
  ARGUMENT_CHECK(initialized_.load(), "The commander has not initialized");

  return context_.local_size;
}

// process rank
int Commander::Rank() const {
  ARGUMENT_CHECK(initialized_.load(), "The commander has not initialized");

  return context_.world_rank;
}

// local rank
int Commander::LocalRank() const {
  ARGUMENT_CHECK(initialized_.load(), "The commander has not initialized");

  return context_.local_rank;
}

// barrier all process
void Commander::Barrier() {
  ARGUMENT_CHECK(initialized_.load(), "The commander has not initialized");

  // (TODO) Is it OK not run in init thread?
  MPI_CALL(MPI_Barrier(context_.world_comm));
}

// local barrier process
void Commander::LocalBarrier() {
  ARGUMENT_CHECK(initialized_.load(), "The commander has not initialized");

  // (TOOD) Is it OK not run in init thread?
  MPI_CALL(MPI_Barrier(context_.local_comm));
}

std::shared_ptr<LockTensor> Commander::CachedLTensor(TaskKey key) const {
  auto it = cached_l_tensors_.find(key);
  if (it == cached_l_tensors_.end()) {
    return nullptr;
  }

  return it->second;
}

void Commander::InsertLTensor(TaskKey key,
                              std::shared_ptr<LockTensor> l_tensor) {
  cached_l_tensors_.emplace(key, l_tensor);
}

// insert a message to message_queue_
void Commander::EnqueueTask(Task&& t) {
  ARGUMENT_CHECK(initialized_.load(), "The commander has not initialized");

  ARGUMENT_CHECK(task_queue_.enqueue(t), "Enqueue task to queue get error!");
}

}  // namespace dadt

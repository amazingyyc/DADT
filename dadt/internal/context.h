#ifndef CONTEXT_H
#define CONTEXT_H

#include <atomic>
#include <mpi.h>

#ifdef HAVE_NCCL
#include <cuda_runtime.h>
#include <nccl.h>
#endif

namespace dadt {

// a config used for communicate with python
struct Config {
  // background thread sleep time, millisecond
  int cycle_duration_ms;

  // what kind broad cast executor used
  // 0: mpi
  // 1: nccl
  int broad_cast_executor_type;

  // what kind all reduce executor should be used
  // 0: mpi all reduce
  // 1: nccl all reduce
  int all_reduce_executor_type;

  // all reduce buffer size
  size_t all_reduce_buffer_size;

  // timeline file path
  const char *timeline_path;
};

// context include the MPI nccl context
struct Context {
  // the MPI word communicator include all process
  // the process size
  // word rank
  MPI_Comm world_comm;
  int world_size;
  int world_rank;

  // communicator in local machine
  // local machine process size
  // local rank
  MPI_Comm local_comm;
  int local_size;
  int local_rank;

  // cross comm, the same index of every process will in the same cross communicator
  MPI_Comm cross_comm;
  int cross_rank;
  int cross_size;

  // rank 0 is leader
  bool is_leader;

  // local rank is 0
  bool is_local_leader;

  // is cross leader
  bool is_cross_leader;

  // create a flaot16 data type for mpi
  MPI_Datatype MPI_FLOAT16_T;

#ifdef HAVE_NCCL
  // gpu_device_id always equal local_rank.
  int gpu_device_id;

  // cuda stream
  cudaStream_t cuda_stream;

  // nccl unique id
  ncclUniqueId nccl_id;

  // nccl comm
  ncclComm_t nccl_comm;
#endif

  // thread cycle duration microsecond
  int64_t cycle_duration_us;

  // whether enable timeline
  std::atomic<bool> enable_timeline;
};

}

#endif
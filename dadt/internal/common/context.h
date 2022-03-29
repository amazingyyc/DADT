#pragma once

#include <mpi.h>

#include <atomic>

#ifdef HAVE_NCCL
#include <cuda_runtime.h>
#include <nccl.h>
#endif

namespace dadt {

// a config used for communicate with python
struct Config {
  // background thread sleep time, millisecond
  uint32_t cycle_duration_ms;

  // what kind broad cast executor used
  // mpi or nccl
  std::string executor_type;

  // all reduce buffer size
  size_t all_reduce_buffer_size;

  // use group_buffer_size to group the tensor
  size_t group_buffer_size;

  // -1 means same with local rank.
  int gpu_device_id{-1};
};

// context include the MPI nccl context
struct Context {
  // the MPI word communicator include all process
  MPI_Comm world_comm;

  // the process size
  int world_size;

  // word rank
  int world_rank;

  // communicator in local machine
  MPI_Comm local_comm;

  // local machine process size
  int local_size;

  // local rank
  int local_rank;

  // cross comm, the same local_rank of every process will in the same cross
  // communicator
  MPI_Comm cross_comm;
  int cross_size;
  int cross_rank;

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
  uint32_t cycle_duration_us;

  // the allreduce buffer size
  size_t all_reduce_buffer_size;

  // use group_buffer_size to group the tensor
  size_t group_buffer_size;
};

}  // namespace dadt

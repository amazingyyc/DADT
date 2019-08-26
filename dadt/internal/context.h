#ifndef CONTEXT_H
#define CONTEXT_H

#include <mpi.h>

#ifdef HAVE_CUDA
#include <cuda_runtime.h>
#include <nccl.h>
#endif

namespace dadt {

// use gradient sum not average
#define DADT_ALLREDUCE_AVERAGE_DISABLE "DADT_ALLREDUCE_AVERAGE_DISABLE"

// cycle time
#define DADT_CYCLE_DURATION_MS "DADT_CYCLE_DURATION_MS"

// context include the MPI context
// and some env
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

  // cross comm, the same index of every machine will in the same cross communicator
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
  
#ifdef HAVE_CUDA
  // cuda stream
  cudaStream_t cuda_stream;

  // nccl unique id
  ncclUniqueId nccl_id;

  // nccl comm
  ncclComm_t nccl_comm;
#endif

  // thread cycle duration milliseconds
  int64_t cycle_duration_ms = 5;
  int64_t cycle_duration_us;
};

}

#endif
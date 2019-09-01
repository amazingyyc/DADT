cmake_minimum_required(VERSION 3.8)

project(dadt LANGUAGES CXX)

set(DADT_HEADER_FILES
  internal/commander.h
  internal/context.h
  internal/device.h
  internal/definition.h
  internal/element_type.h
  internal/lock_tensor.h
  internal/memory_buffer.h
  internal/mpi_all_reduce_executor.h
  internal/mpi_broad_cast_executor.h
  internal/shape.h
  internal/spin_lock.h
  internal/task_executor.h
  internal/task.h
  internal/tensor_storage.h
  internal/tensor.h
  internal/thread_pool.h
  internal/internal.h

  tensorflow/tensorflow_utils.h
)

set(DADT_SRC_FILES
  internal/commander.cc
  internal/device.cc
  internal/lock_tensor.cc
  internal/memory_buffer.cc
  internal/mpi_all_reduce_executor.cc
  internal/mpi_broad_cast_executor.cc
  internal/shape.cc
  internal/spin_lock.cc
  internal/task_executor.cc
  internal/tensor_storage.cc
  internal/tensor.cc
  internal/thread_pool.cc
  internal/internal.cc

  tensorflow/tensorflow_utils.cc
  tensorflow/dadt_broad_cast_op.cc
  tensorflow/dadt_all_reduce_op.cc
)

if (HAVE_CUDA)
  list(APPEND DADT_HEADER_FILES internal/nccl_all_reduce_executor.h)
  list(APPEND DADT_SRC_FILES internal/nccl_all_reduce_executor.cc)
endif()

add_library(dadt SHARED ${DADT_HEADER_FILES} ${DADT_SRC_FILES})

if (HAVE_CUDA)
  target_link_libraries(dadt json11 ${MPI_CXX_LIBRARIES} ${TENSORFLOW_LIB_PATHS} ${CUDA_LIB_PATHS})
else()
  target_link_libraries(dadt json11 ${MPI_CXX_LIBRARIES} ${TENSORFLOW_LIB_PATHS})
endif()

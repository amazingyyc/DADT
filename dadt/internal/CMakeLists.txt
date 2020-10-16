cmake_minimum_required(VERSION 3.5)

# dadt project
project(dadt LANGUAGES CXX)

set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} ${MPI_CXX_COMPILE_FLAGS} ${MPI_CXX_LINK_FLAGS}")

set(DADT_HEADER_FILES
  commander.h
  communicator.h
  context.h
  device.h
  definition.h
  element_type.h
  group.h
  lock_tensor.h
  memory_buffer.h
  mpi_all_reduce_executor.h
  mpi_broad_cast_executor.h
  shape.h
  spin_lock.h
  task_executor.h
  task.h
  tensor_storage.h
  thread_pool.h
  internal.h
  timeline.h
)

set(DADT_SRC_FILES
  commander.cc
  communicator.cc
  device.cc
  group.cc
  lock_tensor.cc
  memory_buffer.cc
  mpi_all_reduce_executor.cc
  mpi_broad_cast_executor.cc
  shape.cc
  spin_lock.cc
  task_executor.cc
  tensor_storage.cc
  thread_pool.cc
  internal.cc
  timeline.cc
)

if (HAVE_NCCL)
  list(APPEND DADT_HEADER_FILES nccl_all_reduce_executor.h)
  list(APPEND DADT_HEADER_FILES nccl_broad_cast_executor.h)
  list(APPEND DADT_HEADER_FILES mpi_cuda_broad_cast_executor.h)
  list(APPEND DADT_HEADER_FILES mpi_cuda_all_reduce_executor.h)

  list(APPEND DADT_SRC_FILES nccl_all_reduce_executor.cc)
  list(APPEND DADT_SRC_FILES nccl_broad_cast_executor.cc)
  list(APPEND DADT_HEADER_FILES mpi_cuda_broad_cast_executor.cc)
  list(APPEND DADT_HEADER_FILES mpi_cuda_all_reduce_executor.cc)
endif()

add_library(dadt SHARED ${DADT_HEADER_FILES} ${DADT_SRC_FILES})

if (HAVE_NCCL)
  target_link_libraries(dadt json11 ${MPI_CXX_LIBRARIES} ${CUDA_LIB_PATHS} ${NCCL_LIB_PATHS})
else()
  target_link_libraries(dadt json11 ${MPI_CXX_LIBRARIES})
endif()

set_target_properties(dadt PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${DADT_LIBRARY_OUTPUT_DIRECTORY})
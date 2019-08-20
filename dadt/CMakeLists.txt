cmake_minimum_required(VERSION 3.8)

project(dadt LANGUAGES CXX)

# c++11
set(CMAKE_CXX_STANDARD 11)

# if have cuda
if (HAVE_CUDA)
  add_definitions(-DHAVE_CUDA)
endif()

set(DADT_HEADER_FILES
  include/internal/commander.h
  include/internal/context.h
  include/internal/device.h
  include/internal/element_type.h
  include/internal/lock_tensor.h
  include/internal/memory_buffer.h
  include/internal/mpi_all_reduce_executor.h
  include/internal/mpi_broad_cast_executor.h
  include/internal/shape.h
	include/internal/spin_lock.h
	include/internal/task_executor.h
	include/internal/task.h
	include/internal/tensor_storage.h
	include/internal/tensor.h
  include/internal/internal.h
)

set(DADT_SRC_FILES
  src/internal/commander.cc
  src/internal/device.cc
  src/internal/lock_tensor.cc
  src/internal/memory_buffer.cc
  src/internal/mpi_all_reduce_executor.cc
  src/internal/mpi_broad_cast_executor.cc
  src/internal/shape.cc
	src/internal/spin_lock.cc
	src/internal/tensor_storage.cc
	src/internal/tensor.cc
  src/internal/internal.cc
)

add_library(dadt ${DADT_HEADER_FILES} ${DADT_SRC_FILES})
target_link_libraries(dadt RapidJSON ${MPI_C_LIBRARIES})
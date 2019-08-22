cmake_minimum_required(VERSION 3.8)

project(dadt LANGUAGES CXX)

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
  include/internal/thread_pool.h
  include/internal/internal.h

  include/tensorflow/tensorflow_utils.h
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
  src/internal/thread_pool.cc
  src/internal/internal.cc

  src/tensorflow/tensorflow_utils.cc
  src/tensorflow/dadt_broad_cast_op.cc
  src/tensorflow/dadt_all_reduce_op.cc
)

add_library(dadt SHARED ${DADT_HEADER_FILES} ${DADT_SRC_FILES})

if (HAVE_CUDA)
  target_link_libraries(dadt json11 ${MPI_C_LIBRARIES} ${TENSORFLOW_LIB_PATHS} ${CUDA_LIB_PATHS})
else()
  target_link_libraries(dadt json11 ${MPI_C_LIBRARIES} ${TENSORFLOW_LIB_PATHS})
endif()

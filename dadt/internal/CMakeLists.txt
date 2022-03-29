cmake_minimum_required(VERSION 3.5)

# dadt project
project(dadt LANGUAGES CXX)

set(CMAKE_CXX_FLAGS
    "${CMAKE_CXX_FLAGS} ${MPI_CXX_COMPILE_FLAGS} ${MPI_CXX_LINK_FLAGS}")

# include dir include_directories(kraken)

file(GLOB_RECURSE DADT_HEADER_FILES "*.h")
file(GLOB_RECURSE DADT_SRC_FILES "*.cc")

set(NCCL_HEADER_FILES
    executor/nccl_all_reduce_executor.h executor/nccl_broad_cast_executor.h
    executor/mpi_cuda_broad_cast_executor.h
    executor/mpi_cuda_all_reduce_executor.h)

set(NCCL_SRC_FILES
    executor/nccl_all_reduce_executor.cc executor/nccl_broad_cast_executor.cc
    executor/mpi_cuda_broad_cast_executor.cc
    executor/mpi_cuda_all_reduce_executor.cc)

if(HAVE_NCCL)
  message(STATUS "Build with NCCL")
else()
  message(STATUS "Build without NCCL")

  list(REMOVE_ITEM DADT_HEADER_FILES ${NCCL_HEADER_FILES})
  list(REMOVE_ITEM DADT_SRC_FILES ${NCCL_SRC_FILES})
endif()

add_library(dadt SHARED ${DADT_HEADER_FILES} ${DADT_SRC_FILES})

if(HAVE_NCCL)
  target_link_libraries(dadt ${MPI_CXX_LIBRARIES} ${CUDA_LIB_PATHS}
                        ${NCCL_LIB_PATHS})
else()
  target_link_libraries(dadt ${MPI_CXX_LIBRARIES})
endif()

cmake_minimum_required(VERSION 3.5)

# c++11
set(CMAKE_CXX_STANDARD 11)

# print debug log
# add_definitions(-DDEBUG_LOG)

# find mpi
find_package(MPI REQUIRED)

# mpi include path
include_directories(SYSTEM ${MPI_CXX_INCLUDE_PATH})

set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} ${MPI_CXX_COMPILE_FLAGS} ${MPI_CXX_LINK_FLAGS}")

# include third_party 
include_directories(third_party/concurrentqueue)
include_directories(third_party/json11)

# add json11 sumodule
add_subdirectory(third_party/json11)

# inlude tensorflow
include_directories(${TENSORFLOW_INCLUDE_DIRS})

# if have cuda
if (HAVE_NCCL)
  include_directories(${CUDA_INCLUDE_DIRS})
  include_directories(${NCCL_INCLUDE_DIRS})
endif()

# dadt include
include_directories(dadt/internal)
include_directories(dadt/tensorflow)

# dadt module
add_subdirectory(dadt)

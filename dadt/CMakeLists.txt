cmake_minimum_required(VERSION 3.5)

# c++14
set(CMAKE_CXX_STANDARD 14)

# set D_GLIBCXX_USE_CXX11_ABI
if (ADD_GLIBCXX_USE_CXX11_ABI)
  add_definitions(-D_GLIBCXX_USE_CXX11_ABI=0)
endif()

# find mpi
find_package(MPI REQUIRED)

# mpi include path
include_directories(SYSTEM ${MPI_CXX_INCLUDE_PATH})

# add nccl definition
if (HAVE_NCCL)
  add_definitions(-DHAVE_NCCL)
endif()

# if have cuda
if (HAVE_NCCL)
  include_directories(${CUDA_INCLUDE_DIRS})
  include_directories(${NCCL_INCLUDE_DIRS})
endif()

# inlude python 
include_directories(${PYTHON_INCLUDE_DIRS})

# include third_party 
include_directories(third_party/concurrentqueue)
include_directories(third_party/json11)

# internal pytorch include
include_directories(internal)
include_directories(pytorch)

# add json11 sumodule
add_subdirectory(third_party/json11)

# add dadt internal sumodule
add_subdirectory(internal)

# add pytroch submodule
add_subdirectory(pytorch)
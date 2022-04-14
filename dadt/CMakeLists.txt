cmake_minimum_required(VERSION 3.5)

project(dadt)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# set D_GLIBCXX_USE_CXX11_ABI
if(ADD_GLIBCXX_USE_CXX11_ABI)
  add_definitions(-D_GLIBCXX_USE_CXX11_ABI=0)
endif()

# Add search folder make sure to find the mpi
set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} "/usr/lib64/openmpi")

# find mpi
find_package(MPI REQUIRED)

# mpi include path
include_directories(SYSTEM ${MPI_CXX_INCLUDE_PATH})

# try to find openmp
find_package(OpenMP)

if(OPENMP_FOUND)
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
  set(CMAKE_EXE_LINKER_FLAGS
      "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")

  # add a OpenMP flag
  add_definitions(-DHAVE_OPENMP)
else()
  message(WARNING "Can not find OpenMP")
endif()

# add nccl definition
if(HAVE_NCCL)
  add_definitions(-DHAVE_NCCL)
endif()

# if have cuda
if(HAVE_NCCL)
  include_directories(${CUDA_INCLUDE_DIRS})
  include_directories(${NCCL_INCLUDE_DIRS})
endif()

# inlude python
include_directories(${PYTHON_INCLUDE_DIRS})

# include third_party
include_directories(third_party/concurrentqueue)

# internal pytorch include
include_directories(internal)
include_directories(pytorch)

# add dadt internal sumodule
add_subdirectory(internal)

# add pytroch submodule
add_subdirectory(pytorch)

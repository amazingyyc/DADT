cmake_minimum_required(VERSION 3.8)

# c++11
set(CMAKE_CXX_STANDARD 11)
add_compile_options(-std=c++11)

# find mpi
if (WIN32)
  if (NOT DEFINED MPI_INCLUDE_PATH)
    set(MPI_INCLUDE_PATH "C:/Program\ Files\ (x86)/Microsoft\ SDKs/MPI/Include")
  endif()

  if (NOT DEFINED MPI_C_LIBRARIES)
    set(MPI_C_LIBRARIES
      "C:/Program\ Files\ (x86)/Microsoft SDKs/MPI/Lib/x64/msmpi.lib"
      "C:/Program\ Files\ (x86)/Microsoft SDKs/MPI/Lib/x64/msmpifec.lib"
      "C:/Program\ Files\ (x86)/Microsoft SDKs/MPI/Lib/x64/msmpifmc.lib")
  endif()
else()
  find_package(MPI REQUIRED)
endif()

# mpi include path
include_directories(SYSTEM ${MPI_INCLUDE_PATH})

# include third_party 
include_directories(third_party)
include_directories(third_party/concurrentqueue)
include_directories(third_party/json11)

# add json11 sumodule
add_subdirectory(third_party/json11)

# inlude tensorflow
include_directories(${TENSORFLOW_INCLUDE_DIRS})

# if have cuda
if (HAVE_CUDA)
  include_directories(${CUDA_INCLUDE_DIRS})
endif()

# dadt include
include_directories(dadt)
include_directories(dadt/include)
include_directories(dadt/include/internal)
include_directories(dadt/include/tensorflow)

# dadt module
add_subdirectory(dadt)

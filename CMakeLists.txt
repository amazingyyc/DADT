cmake_minimum_required(VERSION 3.8)

set(CMAKE_CXX_STANDARD 11)

# if have cuda
set(HAVE_CUDA FALSE)

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

# dadt include
include_directories(dadt)
include_directories(dadt/include)
include_directories(dadt/include/internal)

# dadt module
add_subdirectory(dadt)

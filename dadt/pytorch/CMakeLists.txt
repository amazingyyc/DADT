cmake_minimum_required(VERSION 3.5)

# dadt_pytorch project
project(dadt_pytorch LANGUAGES CXX)

# c++14
set(CMAKE_CXX_STANDARD 14)

# inlude python 
include_directories(${PYTORCH_INCLUDE_DIRS})

add_library(dadt_pytorch SHARED pytorch_utils.cc operators.cc pytorch_tensor.cc)

if (HAVE_NCCL)
  target_link_libraries(dadt_pytorch dadt ${CUDA_LIB_PATHS} ${NCCL_LIB_PATHS} ${PYTORCH_LIB_PATHS})
else()
  target_link_libraries(dadt_pytorch)
endif()

set_target_properties(dadt_pytorch PROPERTIES SUFFIX "${PYTHON_LIB_SUFFIX}")
set_target_properties(dadt_pytorch PROPERTIES PREFIX "")
set_target_properties(dadt_pytorch PROPERTIES OUTPUT_NAME "dadt_pytorch")
set_target_properties(dadt_pytorch PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${DADT_PYTORCH_LIBRARY_OUTPUT_DIRECTORY})
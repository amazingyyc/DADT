cmake_minimum_required(VERSION 3.5)

# dadt_pytorch project
project(dadt_pytorch LANGUAGES CXX)

# inlude python
include_directories(${PYTORCH_INCLUDE_DIRS})

file(GLOB_RECURSE DADT_PYTORCH_HEADER_FILES "*.h")
file(GLOB_RECURSE DADT_PYTORCH_SRC_FILES "*.cc")

add_library(dadt_pytorch SHARED ${DADT_HEADER_FILES} ${DADT_PYTORCH_SRC_FILES})

if(HAVE_NCCL)
  target_link_libraries(dadt_pytorch dadt ${CUDA_LIB_PATHS} ${NCCL_LIB_PATHS}
                        ${PYTORCH_LIB_PATHS})
else()
  target_link_libraries(dadt_pytorch dadt "${PYTORCH_LIB_PATHS}")
endif()

# remove lib prefix for dadt_pytorch
set_target_properties(dadt_pytorch PROPERTIES PREFIX "")

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
  # find_library( PYTORCH_LIBRARY NAMES libc10_cuda.so libc10.so
  # libcaffe2_detectron_ops_gpu.so libcaffe2_module_test_dynamic.so
  # libcaffe2_nvrtc.so libcaffe2_observers.so libcudart-80664282.so.10.2
  # libgomp-a34b3233.so.1 libnvrtc-08c4863f.so.10.2 libnvrtc-builtins.so
  # libnvToolsExt-3965bdd0.so.1 libshm.so libtorch_cpu.so libtorch_cuda.so
  # libtorch_global_deps.so libtorch_python.so libtorch.so HINTS
  # "/usr/local/lib64/python3.6/site-packages/torch/lib/")
  # link_directories("/usr/local/lib64/python3.6/site-packages/torch/lib/")

  target_link_libraries(dadt_pytorch dadt "${PYTORCH_LIB_PATHS}")
  # message(STATUS "PYTORCH_LIBRARY:${PYTORCH_LIBRARY}")
  # find_library(PYTORCH_LIBRARY_0 torch_python
  #              HINTS "/usr/local/lib64/python3.6/site-packages/torch/lib/")
  # find_library(PYTORCH_LIBRARY_1 torch_cpu
  #              HINTS "/usr/local/lib64/python3.6/site-packages/torch/lib/")
  # find_library(PYTORCH_LIBRARY_2 c10_cuda
  #              HINTS "/usr/local/lib64/python3.6/site-packages/torch/lib/")
  # find_library(PYTORCH_LIBRARY_3 c10
  #              HINTS "/usr/local/lib64/python3.6/site-packages/torch/lib/")
  # find_library(PYTORCH_LIBRARY_4 torch_cuda
  #              HINTS "/usr/local/lib64/python3.6/site-packages/torch/lib/")
  # find_library(PYTORCH_LIBRARY_5 torch
  #              HINTS "/usr/local/lib64/python3.6/site-packages/torch/lib/")

  # target_link_libraries(
  #   dadt_pytorch
  #   dadt
  #   ${PYTORCH_LIBRARY_0}
  #   ${PYTORCH_LIBRARY_1}
  #   ${PYTORCH_LIBRARY_2}
  #   ${PYTORCH_LIBRARY_3}
  #   ${PYTORCH_LIBRARY_4}
  #   ${PYTORCH_LIBRARY_5})
endif()

# remove lib prefix for dadt_pytorch
set_target_properties(dadt_pytorch PROPERTIES PREFIX "")

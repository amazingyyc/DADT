#ifndef EXCEPTION_H
#define EXCEPTION_H

#include <iostream>
#include <string>
#include <sstream>

namespace dadt {
  
#define ARGUMENT_CHECK(condition, message)                                     \
  if (!(condition)) {                                                          \
    std::ostringstream oss;                                                    \
    oss << __FILE__ << ":";                                                    \
    oss << __LINE__ << ":";                                                    \
    oss << message << ".";                                                     \
    throw std::invalid_argument(oss.str());                                    \
  }

#define RUNTIME_ERROR(message)                                                 \
  {                                                                            \
    std::ostringstream oss;                                                    \
    oss << __FILE__ << ":";                                                    \
    oss << __LINE__ << ":";                                                    \
    oss << message << ".";                                                     \
    throw std::runtime_error(oss.str());                                       \
  }

#define MPI_CALL(operation)                                                     \
{                                                                               \
  auto mpi_code = (operation);                                                  \
  if (MPI_SUCCESS != mpi_code) {                                                \
    char mpi_error_str[MPI_MAX_ERROR_STRING];                                   \
    int mpi_error_length;                                                       \
    MPI_Error_string(mpi_code, mpi_error_str, &mpi_error_length);               \
    std::ostringstream oss;                                                     \
    oss << __FILE__ << ":";                                                     \
    oss << __LINE__ << ":\n";                                                   \
    oss << "MPI error string:" << std::string(mpi_error_str) << ".";            \
  }                                                                             \
}

#define CUDA_CALL(cudaExecute)                                                  \
{                                                                               \
  auto ret = (cudaExecute);                                                     \
  if (ret != cudaSuccess) {                                                     \
    DEEP8_RUNTIME_ERROR("the CUDA get a error: "                                \
    << #cudaExecute                                                             \
    << ","                                                                      \
    << cudaGetErrorString(ret));                                                \
  }                                                                             \
};                                                                              \

}

#endif
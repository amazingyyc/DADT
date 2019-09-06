#ifndef DEFINITION_H
#define DEFINITION_H

#include <iostream>
#include <string>
#include <sstream>
#include <stdexcept>

namespace dadt {

#ifdef DEBUG_LOG
#define LOG_INFO(rank, info)                                                   \
  {                                                                            \
    std::ostringstream oss;                                                    \
    oss << "rank:" << rank << ", " << info << ".";                             \
    std::cout << oss.str() << "\n";                                            \
  }
#else
#define LOG_INFO(rank, info)
#endif

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
    throw std::runtime_error(oss.str());                                        \
  }                                                                             \
}

#ifdef HAVE_NCCL

#define CUDA_CALL(cudaExecute)                                                  \
{                                                                               \
  auto ret = (cudaExecute);                                                     \
  if (ret != cudaSuccess) {                                                     \
    std::ostringstream oss;                                                     \
    oss << __FILE__ << ":";                                                     \
    oss << __LINE__ << ":\n";                                                   \
    oss << "CUDA get a error: ";                                                \
    oss << cudaGetErrorString(ret);                                             \
    throw std::runtime_error(oss.str());                                        \
  }                                                                             \
};

#define NCCL_CALL(ncclExecute)                                                  \
{                                                                               \
  auto ret = (ncclExecute);                                                     \
  if (ret != ncclSuccess) {                                                     \
    std::ostringstream oss;                                                     \
    oss << __FILE__ << ":";                                                     \
    oss << __LINE__ << ":\n";                                                   \
    oss << "NCCL get a error: ";                                                \
    oss << ncclGetErrorString(ret);                                             \
    throw std::runtime_error(oss.str());                                        \
  }                                                                             \
};

#endif

}

#endif
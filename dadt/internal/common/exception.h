#pragma once

#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>

namespace dadt {

#define ARGUMENT_CHECK(cond, msg) \
  if (!(cond)) { \
    std::ostringstream _oss; \
    _oss << "[" << std::this_thread::get_id() << "] "; \
    _oss << __FILE__ << ":"; \
    _oss << __LINE__ << ":"; \
    _oss << msg << "."; \
    throw std::invalid_argument(_oss.str()); \
  }

#define RUNTIME_ERROR(msg) \
  { \
    std::ostringstream _oss; \
    _oss << "[" << std::this_thread::get_id() << "] "; \
    _oss << __FILE__ << ":"; \
    _oss << __LINE__ << ":"; \
    _oss << msg << "."; \
    throw std::runtime_error(_oss.str()); \
  }

#define MPI_CALL(operation) \
  { \
    auto mpi_code = (operation); \
    if (MPI_SUCCESS != mpi_code) { \
      char mpi_error_str[MPI_MAX_ERROR_STRING]; \
      int mpi_error_length; \
      MPI_Error_string(mpi_code, mpi_error_str, &mpi_error_length); \
      std::ostringstream oss; \
      oss << __FILE__ << ":"; \
      oss << __LINE__ << ":\n"; \
      oss << "MPI error string:" << std::string(mpi_error_str) << "."; \
      throw std::runtime_error(oss.str()); \
    } \
  }

#ifdef HAVE_NCCL
#define CUDA_CALL(cudaExecute) \
  { \
    auto ret = (cudaExecute); \
    if (ret != cudaSuccess) { \
      std::ostringstream oss; \
      oss << __FILE__ << ":"; \
      oss << __LINE__ << ":\n"; \
      oss << "CUDA get error, "; \
      oss << "code:" << ret << ", msg:"; \
      oss << cudaGetErrorString(ret); \
      throw std::runtime_error(oss.str()); \
    } \
  };

#define NCCL_CALL(ncclExecute) \
  { \
    auto ret = (ncclExecute); \
    if (ret != ncclSuccess) { \
      std::ostringstream oss; \
      oss << __FILE__ << ":"; \
      oss << __LINE__ << ":\n"; \
      oss << "NCCL get a error, "; \
      oss << "code:" << ret << ", msg:"; \
      oss << ncclGetErrorString(ret); \
      throw std::runtime_error(oss.str()); \
    } \
  };
#endif

}  // namespace dadt

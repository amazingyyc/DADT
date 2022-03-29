#pragma once

#include <sstream>
#include <thread>

namespace dadt {
namespace log {

extern uint32_t LOG_LEVEL;

class Logger {
private:
  std::ostringstream& oss_;

public:
  Logger(std::ostringstream& oss) : oss_(oss) {
  }

  template <typename T>
  Logger& operator<<(const T& v) {
    oss_ << v;
    return *this;
  }
};

}  // namespace log

#define LOG_DEBUG_LEVEL 0
#define LOG_INFO_LEVEL 1
#define LOG_WARNING_LEVEL 2
#define LOG_ERROR_LEVEL 3

#define REAL_PRINT_LOG(msg) \
  { \
    std::ostringstream _oss; \
    dadt::log::Logger _logger(_oss); \
    _logger << "[" << std::this_thread::get_id() << "] "; \
    _logger << __FILE__ << ":"; \
    _logger << __LINE__ << ":"; \
    _logger << msg << "\n"; \
    std::cout << _oss.str(); \
  }

#define PRINT_LOG(msg, level) \
  if (level >= dadt::log::LOG_LEVEL) { \
    REAL_PRINT_LOG(msg) \
  }

#define LOG_DEBUG(msg) PRINT_LOG(msg, LOG_DEBUG_LEVEL)
#define LOG_INFO(msg) PRINT_LOG(msg, LOG_INFO_LEVEL)
#define LOG_WARNING(msg) PRINT_LOG(msg, LOG_WARNING_LEVEL)
#define LOG_ERROR(msg) PRINT_LOG(msg, LOG_ERROR_LEVEL)

}  // namespace dadt
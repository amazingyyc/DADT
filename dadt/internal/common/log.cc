#include "common/log.h"

namespace dadt {
namespace log {

#ifdef DEBUG
uint32_t LOG_LEVEL = LOG_DEBUG_LEVEL;
#else
uint32_t LOG_LEVEL = LOG_INFO_LEVEL;
#endif

}  // namespace log
}  // namespace dadt

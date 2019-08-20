#include "exception.h"
#include "internal.h"
#include "commander.h"

namespace dadt {

// the commander
Commander commander_;

// initialize dadt
void initialize() {
  commander_.initialize();
}

// id have been initialized
bool initialized() {
  return commander_.initialized();
}

// how many process 
int size() {
  return commander_.size();
}

// how many process in current machine 
int local_size() {
  return commander_.local_size();
}

// the rank of current process
int rank() {
  return commander_.rank();
}

// local rank
int local_rank() {
  return commander_.local_rank();
}

// barrier all process
void barrier() {
  commander_.barrier();
}

// local barrier all process
void local_barrier() {
  commander_.local_barrier();
}


}
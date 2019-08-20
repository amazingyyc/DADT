#ifndef INTERNAL_H
#define INTERNAL_H

namespace dadt {

// initialize dadt
void initialize();

// id have been initialized
bool initialized();

// how many process 
int size();

// how many process in current machine 
int local_size();

// the rank of current process
int rank();

// local rank
int local_rank();

// barrier all process
void barrier();

// local barrier all process
void local_barrier();

}

#endif

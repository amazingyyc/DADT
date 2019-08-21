#ifndef INTERNAL_H
#define INTERNAL_H

#include <iostream>
#include <functional>

#include "task.h"
#include "element_type.h"
#include "lock_tensor.h"

namespace dadt {

// initialize dadt
void init();

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

// get a interim tensor by TaskType
std::shared_ptr<LockTensor> get_interim_tensor(TaskType task_type, 
                                              std::string name, 
                                              std::vector<int> dims, 
                                              ElementType element_type);

// put a task is async queue
void async_task(std::function<void()> &&task);

}

#endif

#ifndef INTERNAL_H
#define INTERNAL_H

#include <iostream>
#include <functional>
#include <memory>
#include <cstring>

#include "task.h"
#include "element_type.h"
#include "lock_tensor.h"

namespace dadt {

extern "C" {

// initialize dadt
void init();

// stop the background thread
void shutdown();

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

// if have a midway tensor corresponding the tasktype
std::shared_ptr<LockTensor> has_midway_tensor(TaskType, std::string);

// get a interim tensor by TaskType
std::shared_ptr<LockTensor> midway_tensor(TaskType, std::string, std::vector<int>, dadt::ElementType);

// put a task in queue
void enqueue_task(Task &&t);

// put a job is async queue
void async_job(std::function<void()> &&job);

}

#endif

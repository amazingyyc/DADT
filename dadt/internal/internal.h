#ifndef INTERNAL_H
#define INTERNAL_H

#include <iostream>
#include <functional>
#include <memory>
#include <cstring>

#include "context.h"
#include "task.h"
#include "element_type.h"
#include "lock_tensor.h"
#include "timeline.h"

namespace dadt {

extern "C" {

// initialize dadt
void init(Config config);

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

// put a task in queue
void enqueue_task(Task &&t);

// put a job is async queue
void enqueue_job(std::function<void()> &&job);

// timeline event begin
void begin_timeline_event(const std::string &name, const std::string &event);

// timeline event end
void end_timeline_event(const std::string &name, const std::string &event);

#ifdef HAVE_NCCL
cudaEvent_t obtain_cuda_event();
#endif

// if have a midway tensor corresponding the tasktype
std::shared_ptr<LockTensor> obtain_midway_tensor(TaskType, std::string);

// get a interim tensor by TaskType
std::shared_ptr<LockTensor> create_midway_tensor(TaskType, std::string, std::vector<int>, ElementType);

}

#endif

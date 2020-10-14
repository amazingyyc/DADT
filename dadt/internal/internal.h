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

// whether have been initialized
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

// put a job into threadpool
void thread_pool_enqueue(std::function<void()> &&task);

// wait all task in thread pool finish
void thread_pool_wait();

// put a task in queue
void enqueue_task(Task &&t);

// timeline event begin
void begin_timeline_event(const std::string &name, const std::string &event);

// timeline event end
void end_timeline_event(const std::string &name, const std::string &event);

#ifdef HAVE_NCCL
cudaEvent_t obtain_cuda_event();
#endif

bool is_cuda_midway_tensor(TaskType task_type);

// insert a midway tensor into excutor
void insert_midway_tensor(TaskType task_type, std::string name, std::shared_ptr<LockTensor> tensor);

// if have a midway tensor corresponding the tasktype
std::shared_ptr<LockTensor> obtain_midway_tensor(TaskType, std::string);

// get a interim tensor by TaskType
std::shared_ptr<LockTensor> create_midway_tensor(TaskType, std::string, Shape, ElementType);

}

#endif

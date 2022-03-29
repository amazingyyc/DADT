#include "executor/mpi_all_reduce_executor.h"

#include "common/device.h"
#include "common/exception.h"

namespace dadt {

MPIAllReduceExecutor::MPIAllReduceExecutor(Device* cpu_device,
                                           size_t buffer_size)
    : buffer_(cpu_device) {
  buffer_.Reserve(buffer_size);
}

// split tasks to MergeUnit
std::vector<MergeUnit> MPIAllReduceExecutor::SplitTasks(
    const std::vector<Task>& tasks, size_t buffer_size) {
  std::vector<MergeUnit> merge_units;

  for (size_t i = 0; i < tasks.size();) {
    if (tasks[i].l_tensor->tensor().NumBytes() >= buffer_size) {
      MergeUnit unit;
      unit.begin = i;
      unit.end = i + 1;

      merge_units.emplace_back(unit);
      i += 1;
    } else {
      size_t cur_size = 0;
      size_t j = i;

      for (; j < tasks.size(); ++j) {
        if (cur_size + tasks[j].l_tensor->tensor().NumBytes() > buffer_size) {
          break;
        }

        cur_size += tasks[j].l_tensor->tensor().NumBytes();
      }

      MergeUnit unit;
      unit.begin = i;
      unit.end = j;

      merge_units.emplace_back(unit);
      i = j;
    }
  }

  return merge_units;
}

void MPIAllReduceExecutor::Do(const Context& context,
                              const std::vector<Task>& tasks) {
  if (tasks.empty()) {
    return;
  }

  auto element_type = tasks[0].l_tensor->tensor().element_type();

  for (const auto& task : tasks) {
    ARGUMENT_CHECK(task.l_tensor->tensor().IsCpu(),
                   "MPIAllReduce only support cpu tensor.");
    ARGUMENT_CHECK(task.l_tensor->tensor().IsContiguous(),
                   "MPIAllReduce need tensor is contiguous.");
    ARGUMENT_CHECK(element_type == task.l_tensor->tensor().element_type(),
                   "MPIAllReduce ElementType must same.");
  }

  auto merge_units = SplitTasks(tasks, buffer_.size());

  for (const auto& unit : merge_units) {
    // before callback.
    for (size_t i = unit.begin; i < unit.end; ++i) {
      if (tasks[i].before) {
        tasks[i].before();
      }
    }

    void* recvbuf = nullptr;
    int count = 0;

    if (unit.begin + 1 == unit.end) {
      recvbuf = tasks[unit.begin].l_tensor->tensor().Ptr();
      count = tasks[unit.begin].l_tensor->tensor().Size();
    } else {
      // copy tensor to buffer
      size_t offset = 0;

      for (size_t i = unit.begin; i < unit.end; ++i) {
        std::memcpy(buffer_.ptr(offset), tasks[i].l_tensor->tensor().Ptr(),
                    tasks[i].l_tensor->tensor().NumBytes());

        offset += tasks[i].l_tensor->tensor().NumBytes();
        count += tasks[i].l_tensor->tensor().Size();
      }

      recvbuf = buffer_.ptr();
    }

    auto mpi_dtype = MpiDataType(context, element_type);

    // do all reduce
    MPI_CALL(MPI_Allreduce(MPI_IN_PLACE, recvbuf, count, mpi_dtype, MPI_SUM,
                           context.world_comm));

    // copy back
    if (unit.begin + 1 != unit.end) {
      size_t offset = 0;

      for (size_t i = unit.begin; i < unit.end; ++i) {
        std::memcpy(tasks[i].l_tensor->tensor().Ptr(), buffer_.ptr(offset),
                    tasks[i].l_tensor->tensor().NumBytes());

        offset += tasks[i].l_tensor->tensor().NumBytes();
      }
    }

    // After callback.
    for (size_t i = unit.begin; i < unit.end; ++i) {
      if (tasks[i].done) {
        tasks[i].done();
      }
    }
  }
}

}  // namespace dadt

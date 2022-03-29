#include "executor/mpi_broad_cast_executor.h"

#include "common/exception.h"

namespace dadt {

MPIBroadCastExecutor::MPIBroadCastExecutor() {
}

void MPIBroadCastExecutor::Do(const Context& context,
                              const std::vector<Task>& tasks) {
  // for broad cast we will broad one by one
  for (const auto& task : tasks) {
    ARGUMENT_CHECK(
        task.l_tensor->tensor().IsCpu() &&
            task.l_tensor->tensor().IsContiguous(),
        "MPIBroadCastExecutor only support CPU tensor and must be continues.");

    // before callback
    if (task.before) {
      task.before();
    }

    void* sendbuf = task.l_tensor->tensor().Ptr();
    int count = task.l_tensor->tensor().Size();

    auto mpi_dtype =
        MpiDataType(context, task.l_tensor->tensor().element_type());

    // broad cast from rank 0
    MPI_CALL(MPI_Bcast(sendbuf, count, mpi_dtype, 0, context.world_comm));

    // finish callback
    if (task.done) {
      task.done();
    }
  }
}

}  // namespace dadt

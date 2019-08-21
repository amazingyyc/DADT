#include "tensorflow_utils.h"
#include "internal.h"

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"

using namespace tensorflow;

// at here add a new op for tensorflow to broadcast the weight
class DadtAllReduceOp: public AsyncOpKernel {
public:
  explicit DadtAllReduceOp(OpKernelConstruction* context) : AsyncOpKernel(context) {
  }

  void ComputeAsync(OpKernelContext* context, DoneCallback done) override {
    if (!dadt::initialized()) {
      OP_REQUIRES(context, false, errors::InvalidArgument("dadt has not been initialized"));
    }

    // get input and output
    auto &input = context->input(0);

    // for now only support float
    OP_REQUIRES(context, DT_FLOAT == input.dtype(), errors::InvalidArgument("dadt only support float");

    // create output
    Tensor* output = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, input.shape(), &output));

    // get name
    auto name = name();

    auto midway_tensor = dadt::has_midway_tensor(name);

    if (nullptr == midway_tensor) {
      auto dims = convert_tensor_shape_to_array(input.shape());
      auto element_type = convert_dtype_to_element_type(input.dtype());

      midway_tensor = dadt::midway_tensor(DADTAllReduce, name, dims, element_type);
    }

    bool is_gpu = is_gpu_conext(context);

    // at async queue to wait tensor finish allreduce
    dadt::async_job([name, &input, output, midway_tensor, is_gpu, done] {
      // wait tensor become WaitForFetch than change it to InFetch
      midway_tensor->wait(LockTensorStatus::WaitForFetch, LockTensorStatus::InFetch);

      // copy data back to output
      if (is_gpu) {
        midway_tensor->copy_to_gpu(output->flat<float>().data());
      } else {
        midway_tensor->copy_to_cpu(output->flat<float>().data());
      }

      // change status from InFetch to InFill
      midway_tensor->wait(LockTensorStatus::InFetch, LockTensorStatus::InFill);

      // copy input to tesnor
      if (is_gpu) {
        midway_tensor->copy_from_gpu(input.flat<float>().data());
      } else {
        midway_tensor->copy_from_cpu(input.flat<float>().data());
      }

      // create allreduce task than put it in task queue
      Task task;
      task.task_type = dadt::DADTAllReduce;
      task.name      = name;
      task.tensor    = midway_tensor;
      
      task.done = [midway_tensor] {
        midway_tensor.wait(LockTensorStatus::InExecute, LockTensorStatus::WaitForFetch);
      };

      // change status from InFill to InExecute
      midway_tensor->wait(LockTensorStatus::InFill, LockTensorStatus::InExecute);

      // put task in queue
      dadt::enqueue(task);

      // say to tensorflow have finish the op
      done();
    });  
  }
};

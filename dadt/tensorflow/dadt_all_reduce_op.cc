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

    // create output
    Tensor* output = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, input.shape(), &output));

    // get name
    auto midway_name = name();

    auto midway_tensor = dadt::have_midway_tensor(dadt::DADTAllReduceTaskType, midway_name);

    if (nullptr == midway_tensor) {
      auto dims = convert_tensor_shape_to_array(input.shape());
      auto element_type = convert_dtype_to_element_type(input.dtype());

      midway_tensor = dadt::create_midway_tensor(dadt::DADTAllReduceTaskType, midway_name, dims, element_type);
    }

    bool is_gpu = is_gpu_conext(context);

    // at async queue to wait tensor finish allreduce
    dadt::async_job([midway_name, &input, output, midway_tensor, is_gpu, done] {
      // wait tensor become WaitForFetch than change it to InFetch
      midway_tensor->wait(dadt::LockTensorStatus::WaitForFetch, dadt::LockTensorStatus::InFetch);

      // copy data back to output
      // memory copy in tensorflow op always sync
      midway_tensor->copy_to((void*) output->tensor_data().data(), is_gpu);

      // change status from InFetch to InFill
      midway_tensor->wait(dadt::LockTensorStatus::InFetch, dadt::LockTensorStatus::InFill);

      // copy input to tesnor
      // memory copy in tensorflow op always sync
      midway_tensor->copy_from(input.tensor_data().data(), is_gpu);

      // create allreduce task than put it in task queue
      dadt::Task task;
      task.task_type = dadt::DADTAllReduceTaskType;
      task.name      = midway_name;
      task.tensor    = midway_tensor;
      
      task.done = [midway_tensor] {
        midway_tensor->wait(dadt::LockTensorStatus::InExecute, dadt::LockTensorStatus::WaitForFetch);
      };

      // change status from InFill to InExecute
      midway_tensor->wait(dadt::LockTensorStatus::InFill, dadt::LockTensorStatus::InExecute);

      // put task in queue
      dadt::enqueue_task(std::move(task));

      // tell tensorflow have finish the op
      done();
    });
  }
};

// register to Device
REGISTER_KERNEL_BUILDER(Name("DadtAllReduce").Device(DEVICE_CPU), DadtAllReduceOp);

#ifdef HAVE_NCCL
REGISTER_KERNEL_BUILDER(Name("DadtAllReduce").Device(DEVICE_GPU), DadtAllReduceOp);
#endif

REGISTER_OP("DadtAllReduce")
    .Attr("T: {float16, float32, float64}")
    .Input("input: T")
    .Output("output: T")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      c->set_output(0, c->input(0));
      return Status::OK();
    })
    .Doc(R"doc(
        a dadt all reduce op, will all reduce the input with other rank.
        )doc");


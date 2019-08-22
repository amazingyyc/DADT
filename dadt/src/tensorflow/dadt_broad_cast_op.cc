#include "tensorflow_utils.h"
#include "internal.h"

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"

using namespace tensorflow;

// at here add a new op for tensorflow to broadcast the weight
class DadtBroadCastOp: public AsyncOpKernel {
public:
  explicit DadtBroadCastOp(OpKernelConstruction* context) : AsyncOpKernel(context) {
  }

  void ComputeAsync(OpKernelContext* context, DoneCallback done) override {
    if (!dadt::initialized()) {
      OP_REQUIRES(context, false, errors::InvalidArgument("dadt has not been initialized"));
    }

    // get input
    auto &input = context->input(0);

    // for now only sypport float
    OP_REQUIRES(context, DT_FLOAT == input.dtype(), errors::InvalidArgument("dadt only support float"));

    // create output
    Tensor* output = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, input.shape(), &output));

    // get a interim tensor
    auto midway_name = name();
    auto dims        = convert_tensor_shape_to_array(input.shape());
    auto element_type = convert_dtype_to_element_type(input.dtype());

    // get the interim tensor
    auto tensor = dadt::midway_tensor(dadt::DADTBroadCast, midway_name, dims, element_type);

    // copy input to interim tensor
    bool is_gpu = is_gpu_conext(context);

    if (is_gpu) {
      tensor->copy_from_gpu(input.flat<float>().data());
    } else {
      tensor->copy_from_cpu(input.flat<float>().data());
    }

    // create a task
    dadt::Task task;
    task.task_type = dadt::DADTBroadCast;
    task.name      = midway_name;
    task.tensor    = tensor;

    task.done = [is_gpu, output, tensor, done] {
      // after broadcast should copy data to output
      if (is_gpu) {
        tensor->copy_to_gpu(output->flat<float>().data());
      } else {
        tensor->copy_to_cpu(output->flat<float>().data());
      }

      done();
    };

    // put task in queue
    dadt::enqueue_task(std::move(task));
  }
};

// register to Device
REGISTER_KERNEL_BUILDER(Name("DadtBroadCast").Device(DEVICE_CPU), DadtBroadCastOp);

#ifdef HAVE_CUDA
REGISTER_KERNEL_BUILDER(Name("DadtBroadCast").Device(DEVICE_GPU), DadtBroadCastOp);
#endif

REGISTER_OP("DadtBroadCast")
    .Input("input: float")
    .Output("output: float")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      c->set_output(0, c->input(0));
      return Status::OK();
    })
    .Doc(R"doc(
        a dadt broad cast op, rank 0 will broad cast the input to other rank and put result in output.
        )doc");

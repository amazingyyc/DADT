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

    // create output
    Tensor* output = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, input.shape(), &output));
  }
};

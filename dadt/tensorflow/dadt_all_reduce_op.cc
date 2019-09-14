#ifdef HAVE_NCCL
#define EIGEN_USE_GPU
#endif

#include "tensorflow_utils.h"
#include "internal.h"

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"

using namespace tensorflow;

// all reduce cpu op
class DadtAllReduceOpCPU: public AsyncOpKernel {
public:
  explicit DadtAllReduceOpCPU(OpKernelConstruction* context) : AsyncOpKernel(context) {
  }

  void ComputeAsync(OpKernelContext* context, DoneCallback done) override {
    // get input and output
    auto &input = context->input(0);

    // create output
    Tensor* output = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, input.shape(), &output));

    // get name
    auto op_name = name();

    // get midway tensor
    auto midway_tensor = dadt::obtain_midway_tensor(dadt::DADTAllReduceTaskType, op_name);

    if (nullptr == midway_tensor) {
      auto dims = convert_tensor_shape_to_array(input.shape());
      auto element_type = convert_dtype_to_element_type(input.dtype());

      midway_tensor = dadt::create_midway_tensor(dadt::DADTAllReduceTaskType, op_name, dims, element_type);
    }

    // CPU op only support CPU tensor 
    ARGUMENT_CHECK(dadt::DeviceType::CPU == midway_tensor->device()->device_type(), 
    "CPU op must use CPU tensor, so please set all reduce executor to be MPI");

    // wait midway tensor finish task
    midway_tensor->wait(dadt::LockTensorStatus::WaitForFetch, dadt::LockTensorStatus::InFetch);

    // copy to output
    std::memcpy((void*) output->tensor_data().data(), midway_tensor->ptr(), midway_tensor->num_bytes());

    // copy input to tensor
    std::memcpy(midway_tensor->ptr(), input.tensor_data().data(), midway_tensor->num_bytes());

    // for now the midway result has been copy to output and input has copy in midway tesnor
    // when copy finish create a task put into task queue to do all resuce
    dadt::Task task;
    task.name = op_name;
    task.tensor = midway_tensor;
    task.task_type = dadt::DADTAllReduceTaskType;

    task.done = [midway_tensor] {
      midway_tensor->wait(dadt::LockTensorStatus::InExecute, dadt::LockTensorStatus::WaitForFetch);
    };

    // chang tensor status
    midway_tensor->wait(dadt::LockTensorStatus::InFetch, dadt::LockTensorStatus::InExecute);

    // put task in queue
    dadt::enqueue_task(std::move(task));

    // tell tensorflow have finish the op
    done();
  }
};


#ifdef HAVE_NCCL

// at here add a new op for tensorflow to broadcast the weight
class DadtAllReduceOpGPU: public AsyncOpKernel {
public:
  explicit DadtAllReduceOpGPU(OpKernelConstruction* context) : AsyncOpKernel(context) {
  }

  void ComputeAsync(OpKernelContext* context, DoneCallback done) override {
    // get input and output
    auto &input = context->input(0);

    // create output
    Tensor* output = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, input.shape(), &output));

    // get name
    auto op_name = name();

    // get gpudevice
    const Eigen::GpuDevice& gpu_device = context->eigen_device<Eigen::GpuDevice>();

    auto midway_tensor = dadt::obtain_midway_tensor(dadt::DADTAllReduceTaskType, op_name);

    if (nullptr == midway_tensor) {
      auto dims = convert_tensor_shape_to_array(input.shape());
      auto element_type = convert_dtype_to_element_type(input.dtype());

      midway_tensor = dadt::create_midway_tensor(dadt::DADTAllReduceTaskType, op_name, dims, element_type);
    }

    // wait midway tensor finish task
    midway_tensor->wait(dadt::LockTensorStatus::WaitForFetch, dadt::LockTensorStatus::InFetch);

    // check the midway tensor type
    if (dadt::DeviceType::CPU == midway_tensor->device()->device_type()) {
      // copy memory from cpu tensor to output
      gpu_device.memcpyHostToDevice((void*) output->tensor_data().data(), midway_tensor->ptr(), midway_tensor->num_bytes());

      // copy input to cpu tensor
      gpu_device.memcpyDeviceToHost(midway_tensor->ptr(), input.tensor_data().data(), midway_tensor->num_bytes());
    } else {
      // copy from gpu to gpu
      gpu_device.memcpy((void*) output->tensor_data().data(), midway_tensor->ptr(), midway_tensor->num_bytes());

      // copy input to gpu tensor
      gpu_device.memcpy(midway_tensor->ptr(), input.tensor_data().data(), midway_tensor->num_bytes());
    }

    // wait memory copy finish
    auto wait_event = dadt::obtain_cuda_event();

    // put wait event into stream and wait event finish
    CUDA_CALL(cudaEventRecord(wait_event, gpu_device.stream()));
    CUDA_CALL(cudaEventSynchronize(wait_event));

    // for now the midway result has been copy to output and input has copy in midway tesnor
    // when copy finish create a task put into task queue to do all reduce
    dadt::Task task;
    task.name = op_name;
    task.tensor = midway_tensor;
    task.task_type = dadt::DADTAllReduceTaskType;

    task.done = [midway_tensor] {
      midway_tensor->wait(dadt::LockTensorStatus::InExecute, dadt::LockTensorStatus::WaitForFetch);
    };

    // change tensor status
    midway_tensor->wait(dadt::LockTensorStatus::InFetch, dadt::LockTensorStatus::InExecute);

    // put task in queue
    dadt::enqueue_task(std::move(task));

    // tell tensorflow have finish the op
    done();
  }
};

#endif

// register to Device
REGISTER_KERNEL_BUILDER(Name("DadtAllReduce").Device(DEVICE_CPU), DadtAllReduceOpCPU);

#ifdef HAVE_NCCL
REGISTER_KERNEL_BUILDER(Name("DadtAllReduce").Device(DEVICE_GPU), DadtAllReduceOpGPU);
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


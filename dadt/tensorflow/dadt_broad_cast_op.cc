#ifdef HAVE_NCCL
#define EIGEN_USE_GPU
#endif

#include "tensorflow_utils.h"
#include "internal.h"

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"

using namespace tensorflow;

// at here add a new op for tensorflow to broadcast the weight
class DadtBroadCastOpCPU: public AsyncOpKernel {
public:
  explicit DadtBroadCastOpCPU(OpKernelConstruction* context) : AsyncOpKernel(context) {
  }

  void ComputeAsync(OpKernelContext* context, DoneCallback done) override {
    // get input
    auto &input = context->input(0);

    // create output
    Tensor* output = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, input.shape(), &output));

    // get name
    auto op_name = name();

    auto dims = convert_tensor_shape_to_array(input.shape());
    auto element_type = convert_dtype_to_element_type(input.dtype());

    // broad cast tensor not need reuse
    auto midway_tensor = dadt::create_midway_tensor(dadt::kDADTBroadCastTaskType, op_name, dims, element_type);

    // CPU op only support CPU tensor 
    ARGUMENT_CHECK(dadt::DeviceType::CPU == midway_tensor->device()->device_type(), 
    "CPU op must use CPU tensor, so please set broad cast executor to be MPI");

    // kCopyToMidWayEvent begin
    dadt::begin_timeline_event(op_name, dadt::kCopyToMidWayEvent);

    // copy input to midway tensor
    std::memcpy(midway_tensor->ptr(), input.tensor_data().data(), midway_tensor->num_bytes());

    // kCopyToMidWayEvent begin
    dadt::end_timeline_event(op_name, dadt::kCopyToMidWayEvent);

    // create a task
    dadt::Task task;
    task.name = op_name;
    task.tensor = midway_tensor;
    task.task_type = dadt::kDADTBroadCastTaskType;

    task.done = [output, midway_tensor, done] {
      // after broadcast should copy data to output
      std::memcpy((void*) output->tensor_data().data(), midway_tensor->ptr(), midway_tensor->num_bytes());

      // done callback
      done();
    };

    // put task in queue
    dadt::enqueue_task(std::move(task));
  }
};

#ifdef HAVE_NCCL
class DadtBroadCastOpGPU: public AsyncOpKernel {
public:
  explicit DadtBroadCastOpGPU(OpKernelConstruction* context) : AsyncOpKernel(context) {
  }

  void ComputeAsync(OpKernelContext* context, DoneCallback done) override {
    // get input
    auto &input = context->input(0);

    // create output
    Tensor* output = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, input.shape(), &output));

    // get name
    auto op_name = name();

    // get gpudevice
    const Eigen::GpuDevice& gpu_device = context->eigen_device<Eigen::GpuDevice>();

    auto dims = convert_tensor_shape_to_array(input.shape());
    auto element_type = convert_dtype_to_element_type(input.dtype());

    auto midway_tensor = dadt::create_midway_tensor(dadt::kDADTBroadCastTaskType, op_name, dims, element_type);
    
    // kCopyToMidWayEvent begin
    dadt::begin_timeline_event(op_name, kCopyToMidWayEvent);

    // copy input to tensor
    if (dadt::DeviceType::CPU == midway_tensor->device()->device_type()) {
      // copy input to cpu tensor
      gpu_device.memcpyDeviceToHost(midway_tensor->ptr(), input.tensor_data().data(), midway_tensor->num_bytes());
    } else {
      // copy input to gpu tensor
      gpu_device.memcpy(midway_tensor->ptr(), input.tensor_data().data(), midway_tensor->num_bytes());
    }

    // wait copy finish
    auto wait_event = dadt::obtain_cuda_event();

    // put wait event into stream and wait event finish
    CUDA_CALL(cudaEventRecord(wait_event, gpu_device.stream()));
    CUDA_CALL(cudaEventSynchronize(wait_event));

    // kCopyToMidWayEvent begin
    dadt::end_timeline_event(op_name, kCopyToMidWayEvent);

    // create task
    dadt::Task task;
    task.name = op_name;
    task.tensor = midway_tensor;
    task.task_type = dadt::kDADTBroadCastTaskType;

    task.done = [output, midway_tensor, &gpu_device, wait_event, done] {
      // after broadcast should copy data to output
      if (dadt::DeviceType::CPU == midway_tensor->device()->device_type()) {
        gpu_device.memcpyHostToDevice((void*) output->tensor_data().data(), midway_tensor->ptr(), midway_tensor->num_bytes());
      } else {
        gpu_device.memcpy((void*) output->tensor_data().data(), midway_tensor->ptr(), midway_tensor->num_bytes());
      }

      // wait copy finish
      CUDA_CALL(cudaEventRecord(wait_event, gpu_device.stream()));
      CUDA_CALL(cudaEventSynchronize(wait_event));

      // done callback
      done();
    };

    // put task in queue
    dadt::enqueue_task(std::move(task));
  }
};
#endif

// register to Device
REGISTER_KERNEL_BUILDER(Name("DadtBroadCast").Device(DEVICE_CPU), DadtBroadCastOpCPU);

#ifdef HAVE_NCCL
REGISTER_KERNEL_BUILDER(Name("DadtBroadCast").Device(DEVICE_GPU), DadtBroadCastOpGPU);
#endif

REGISTER_OP("DadtBroadCast")
    .Attr("T: {bool, uint8, int8, uint16, int16, int32, int64, float16, float32, float64}")
    .Input("input: T")
    .Output("output: T")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      c->set_output(0, c->input(0));
      return Status::OK();
    })
    .Doc(R"doc(
        a dadt broad cast op, rank 0 will broad cast the input to other rank and put result in output.
        )doc");

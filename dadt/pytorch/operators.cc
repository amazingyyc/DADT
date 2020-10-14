#include <torch/extension.h>
#include <torch/torch.h>

#ifdef HAVE_NCCL
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>
#endif

#include "internal.h"
#include "lock_tensor.h"
#include "pytorch_tensor.h"
#include "pytorch_utils.h"

namespace dadt {
namespace pytorch {

// Tensor broad_cast CPU, will broad_cast tensor from rank-0 to other ranks.
torch::Tensor broad_cast_cpu(torch::Tensor input, const std::string &name) {
  auto shape = get_shape_vector(input);
  auto element_type = get_element_type(input);

  // broad cast tensor not need reuse
  auto midway_tensor = dadt::create_midway_tensor(dadt::kBroadCastTaskType, name, shape, element_type);

  // CPU op only support CPU tensor
  ARGUMENT_CHECK(midway_tensor->is_cpu(),
    "CPU broadcast must use CPU tensor, so please choose MPI executor to do braodcast.");

  // copy input to midway tensor
  std::memcpy(midway_tensor->ptr(), input.data_ptr(), midway_tensor->num_bytes());

  // change midway_tensor from kInFetch to kInExecute
  midway_tensor->wait(dadt::LockTensorStatus::kInFetch, dadt::LockTensorStatus::kInExecute);

  // create a task
  dadt::Task task;
  task.name = name;
  task.tensor = midway_tensor;
  task.task_type = dadt::kBroadCastTaskType;
  task.done = [midway_tensor] {
    // When finish, change the status.
    midway_tensor->wait(dadt::LockTensorStatus::kInExecute, dadt::LockTensorStatus::kWaitForFetch);
  };

  // put task in queue.
  dadt::enqueue_task(std::move(task));

  // Wait task finish
  midway_tensor->wait(dadt::LockTensorStatus::kWaitForFetch, dadt::LockTensorStatus::kInFetch);

  // Copy result to output.
  auto output = torch::empty_like(input);

  std::memcpy(output.data_ptr(), midway_tensor->ptr(), midway_tensor->num_bytes());

  return output;
}

#ifdef HAVE_NCCL
torch::Tensor broad_cast_gpu(torch::Tensor input, const std::string &name) {
  // Get current cuda stream.
  auto cuda_stream = c10::cuda::getCurrentCUDAStream(input.device().index());

  auto shape = get_shape_vector(input);
  auto element_type = get_element_type(input);

  // broad cast tensor not need reuse
  auto midway_tensor = dadt::create_midway_tensor(dadt::kBroadCastTaskType, name, shape, element_type);

  // copy input to tensor
  if (midway_tensor->is_cpu()) {
    // Copy GPU memory to CPU
    CUDA_CALL(cudaMemcpyAsync(midway_tensor->ptr(),
                              input.data_ptr(),
                              midway_tensor->num_bytes(),
                              cudaMemcpyDeviceToHost,
                              cuda_stream.stream()));
  } else {
    // Copy GPU memory to GPU
    //ARGUMENT_CHECK(input.device().index() == midway_tensor->device()->device_id(),
    //  "Pytorch GPU device index is not same with DADT");

    CUDA_CALL(cudaMemcpyAsync(midway_tensor->ptr(),
                              input.data_ptr(),
                              midway_tensor->num_bytes(),
                              cudaMemcpyDeviceToDevice,
                              cuda_stream.stream()));
  }

  // Wait memory copy finish
  auto wait_event = dadt::obtain_cuda_event();

  // put wait event into stream and wait event finish
  CUDA_CALL(cudaEventRecord(wait_event, cuda_stream.stream()));
  CUDA_CALL(cudaEventSynchronize(wait_event));

  // change midway_tensor from kInFetch to kInExecute
  midway_tensor->wait(dadt::LockTensorStatus::kInFetch, dadt::LockTensorStatus::kInExecute);

  // create a task
  dadt::Task task;
  task.name = name;
  task.tensor = midway_tensor;
  task.task_type = dadt::kBroadCastTaskType;
  task.done = [midway_tensor] {
    // When finish, change the status.
    midway_tensor->wait(dadt::LockTensorStatus::kInExecute, dadt::LockTensorStatus::kWaitForFetch);
  };

  // put task in queue.
  dadt::enqueue_task(std::move(task));

  // Wait task finish
  midway_tensor->wait(dadt::LockTensorStatus::kWaitForFetch, dadt::LockTensorStatus::kInFetch);

  // Copy to output
  auto output = torch::empty_like(input);

  if (midway_tensor->is_cpu()) {
    // Copy CPU memory to GPU
    CUDA_CALL(cudaMemcpyAsync(output.data_ptr(),
                              midway_tensor->ptr(),
                              midway_tensor->num_bytes(),
                              cudaMemcpyHostToDevice,
                              cuda_stream.stream()));
  } else {
    //ARGUMENT_CHECK(output.device().index() == midway_tensor->device()->device_id(),
    //  "Pytorch GPU device index is not same with DADT");

    CUDA_CALL(cudaMemcpyAsync(output.data_ptr(),
                              midway_tensor->ptr(),
                              midway_tensor->num_bytes(),
                              cudaMemcpyDeviceToDevice,
                              cuda_stream.stream()));
  }

  // wait copy to output finish
  CUDA_CALL(cudaEventRecord(wait_event, cuda_stream.stream()));
  CUDA_CALL(cudaEventSynchronize(wait_event));

  return output;
}
#endif

torch::Tensor broad_cast(torch::Tensor input, const std::string &name) {
  if (input.is_cuda()) {
#ifdef HAVE_NCCL
    return broad_cast_gpu(input, name);
#else
    RUNTIME_ERROR("dadt not build with GPU, please rebuild it with GPU.")
#endif
  } else {
    return broad_cast_cpu(input, name);
  }
}

// AllReduce CPU
torch::Tensor all_reduce_cpu(torch::Tensor input, const std::string &name, float multiplier) {
  // create output
  auto output = torch::empty_like(input);

  // get midway tensor
  auto midway_tensor = dadt::obtain_midway_tensor(dadt::kAllReduceTaskType, name);

  if (nullptr == midway_tensor) {
    auto shape = get_shape_vector(input);
    auto element_type = get_element_type(input);

    midway_tensor = dadt::create_midway_tensor(dadt::kAllReduceTaskType, name, shape, element_type);
  }

  // CPU op only support CPU tensor 
  ARGUMENT_CHECK(midway_tensor->is_cpu(), 
    "CPU op must use CPU tensor, choose a MPI executor to do CPU all reduce.");

  // wait midway tensor finish task
  midway_tensor->wait(dadt::LockTensorStatus::kWaitForFetch, dadt::LockTensorStatus::kInFetch);

  // copy to output
  std::memcpy((void*) output.data_ptr(), midway_tensor->ptr(), midway_tensor->num_bytes());

  // copy input to midway_tensor
  std::memcpy(midway_tensor->ptr(), input.data_ptr(), midway_tensor->num_bytes());

  // for now the midway result has been copy to output and input has copy in midway tesnor
  // when copy finish create a task put into task queue to do all reduce
  dadt::Task task;
  task.name = name;
  task.tensor = midway_tensor;
  task.task_type = dadt::kAllReduceTaskType;

  task.done = [midway_tensor] {
    midway_tensor->wait(dadt::LockTensorStatus::kInExecute, dadt::LockTensorStatus::kWaitForFetch);
  };

  // chang tensor status
  midway_tensor->wait(dadt::LockTensorStatus::kInFetch, dadt::LockTensorStatus::kInExecute);

  // put task in queue
  dadt::enqueue_task(std::move(task));

  return output * multiplier;
}

#ifdef HAVE_NCCL
// AllReduce GPU
torch::Tensor all_reduce_gpu(torch::Tensor input, const std::string &name, float multiplier) {
  // Get current cuda stream.
  auto cuda_stream = c10::cuda::getCurrentCUDAStream(input.device().index());

  // create output
  auto output = torch::empty_like(input);

  // get midway tensor
  auto midway_tensor = dadt::obtain_midway_tensor(dadt::kAllReduceTaskType, name);

  if (nullptr == midway_tensor) {
    auto shape = get_shape_vector(input);
    auto element_type = get_element_type(input);

    midway_tensor = dadt::create_midway_tensor(dadt::kAllReduceTaskType, name, shape, element_type);
  }

  // wait midway tensor finish task
  midway_tensor->wait(dadt::LockTensorStatus::kWaitForFetch, dadt::LockTensorStatus::kInFetch);

  // check the midway tensor type
  if (midway_tensor->is_cpu()) {
    // copy memory from cpu tensor to output
    CUDA_CALL(cudaMemcpyAsync(output.data_ptr(),
                              midway_tensor->ptr(),
                              midway_tensor->num_bytes(),
                              cudaMemcpyHostToDevice,
                              cuda_stream.stream()));

    // copy input to cpu tensor
    CUDA_CALL(cudaMemcpyAsync(midway_tensor->ptr(),
                              input.data_ptr(),
                              midway_tensor->num_bytes(),
                              cudaMemcpyDeviceToHost,
                              cuda_stream.stream()));
  } else {
    // copy memory from gpu tensor to output
    CUDA_CALL(cudaMemcpyAsync(output.data_ptr(),
                              midway_tensor->ptr(),
                              midway_tensor->num_bytes(),
                              cudaMemcpyDeviceToDevice,
                              cuda_stream.stream()));

    // copy input to gpu tensor
    CUDA_CALL(cudaMemcpyAsync(midway_tensor->ptr(),
                              input.data_ptr(),
                              midway_tensor->num_bytes(),
                              cudaMemcpyDeviceToDevice,
                              cuda_stream.stream()));
  }

  // wait memory copy finish
  auto wait_event = dadt::obtain_cuda_event();

  // put wait event into stream and wait event finish
  CUDA_CALL(cudaEventRecord(wait_event, cuda_stream.stream()));
  CUDA_CALL(cudaEventSynchronize(wait_event));

  // for now the midway result has been copy to output and input has copy in midway tesnor
  // when copy finish create a task put into task queue to do all reduce
  dadt::Task task;
  task.name = name;
  task.tensor = midway_tensor;
  task.task_type = dadt::kAllReduceTaskType;

  task.done = [midway_tensor] {
    midway_tensor->wait(dadt::LockTensorStatus::kInExecute, dadt::LockTensorStatus::kWaitForFetch);
  };

  // change tensor status
  midway_tensor->wait(dadt::LockTensorStatus::kInFetch, dadt::LockTensorStatus::kInExecute);

  // put task in queue
  dadt::enqueue_task(std::move(task));

  return output * multiplier;
}

// pytorch is trainning on GPU and the midway tensor is GPU too
torch::Tensor all_reduce_gpu_midway_tensor_gpu(torch::Tensor input, const std::string &name, float multiplier) {
  // Get current cuda stream from pytorch.
  auto cuda_stream = c10::cuda::getCurrentCUDAStream(input.device().index()).stream();

  // create a CUDA event
  cudaEvent_t wait_input_event;
  CUDA_CALL(cudaEventCreate(&wait_input_event));

  // put wait event into stream, when the event finish means the input's memory has been ready.
  CUDA_CALL(cudaEventRecord(wait_input_event, cuda_stream));

  // try to get midway tensor maybe get nullptr
  auto midway_tensor = dadt::obtain_midway_tensor(dadt::kAllReduceTaskType, name);

  if (nullptr == midway_tensor) {
    // if midway_tensor is empty means first time. will create a zero tensor.
    auto pytorch_tensor = torch::zeros_like(input);

    // create a midway tensor and store it
    midway_tensor = std::make_shared<PytorchTensor>(pytorch_tensor, name, dadt::LockTensorStatus::kWaitForFetch);

    // store
    dadt::insert_midway_tensor(dadt::kAllReduceTaskType, name, midway_tensor);
  }

  // get pytorch tensor pointer
  PytorchTensor *midway_tensor_ptr = dynamic_cast<PytorchTensor*>(midway_tensor.get());

  // wait midway tensor finish allreduce
  midway_tensor->wait(dadt::LockTensorStatus::kWaitForFetch, dadt::LockTensorStatus::kInFetch);

  // get output
  auto output = midway_tensor_ptr->torch_tensor();

  // put input to midway tensor
  // at here maybe the input still not finish backward, it means the input memory maybe dirty
  // we need wait input finish before do allreduce on input
  midway_tensor_ptr->torch_tensor(input);

  // for now the midway result has been copy to output and input has copy in midway tensor
  // when copy finish create a task put into task queue to do all reduce
  dadt::Task task;
  task.name = name;
  task.tensor = midway_tensor;
  task.task_type = dadt::kAllReduceTaskType;

  task.before = [wait_input_event]() {
    // before use input we need wait input finish backward.
    CUDA_CALL(cudaEventSynchronize(wait_input_event));

    // destroy cuda event
    CUDA_CALL(cudaEventDestroy(wait_input_event));
  };

  task.done = [midway_tensor]() {
    midway_tensor->wait(dadt::LockTensorStatus::kInExecute, dadt::LockTensorStatus::kWaitForFetch);
  };

  // change tensor status
  midway_tensor->wait(dadt::LockTensorStatus::kInFetch, dadt::LockTensorStatus::kInExecute);

  // put task in queue
  dadt::enqueue_task(std::move(task));

  return output * multiplier;
}

#endif

torch::Tensor all_reduce(torch::Tensor input, const std::string &name, float multiplier) {
  if (input.is_cuda()) {
#ifdef HAVE_NCCL
    if (is_cuda_midway_tensor(dadt::kAllReduceTaskType)) {
      // need cuda midway tensor
      return all_reduce_gpu_midway_tensor_gpu(input, name, multiplier);
    } else {
      return all_reduce_gpu(input, name, multiplier);
    }
#else
    RUNTIME_ERROR("dadt not build with GPU, please rebuild it with GPU.")
#endif
  } else {
    return all_reduce_cpu(input, name, multiplier);
  }
}

#ifdef HAVE_NCCL
// AllReduce GPU async, will put the waiting and copy into threadpool
void all_reduce_gpu_async(
  torch::Tensor input,
  torch::Tensor output,
  const std::string &name,
  float multiplier) {
  // Get current cuda stream.
  auto cuda_stream = c10::cuda::getCurrentCUDAStream(input.device().index()).stream();

  // waiting/copy will execute in a threadpool
  auto job = [=]() {
    // get midway tensor
    auto midway_tensor = dadt::obtain_midway_tensor(dadt::kAllReduceTaskType, name);

    if (nullptr == midway_tensor) {
      auto dims = get_shape_vector(input);
      auto element_type = get_element_type(input);

      midway_tensor = dadt::create_midway_tensor(dadt::kAllReduceTaskType, name, dims, element_type);
    }

    // wait midway tensor finish allreduce
    midway_tensor->wait(dadt::LockTensorStatus::kWaitForFetch, dadt::LockTensorStatus::kInFetch);

    // check the midway tensor type
    if (midway_tensor->is_cpu()) {
      // copy memory from cpu tensor to output
      CUDA_CALL(cudaMemcpyAsync(output.data_ptr(),
                                midway_tensor->ptr(),
                                midway_tensor->num_bytes(),
                                cudaMemcpyHostToDevice,
                                cuda_stream));

      // copy input to cpu tensor
      CUDA_CALL(cudaMemcpyAsync(midway_tensor->ptr(),
                                input.data_ptr(),
                                midway_tensor->num_bytes(),
                                cudaMemcpyDeviceToHost,
                                cuda_stream));
    } else {
      // copy memory from gpu tensor to output
      CUDA_CALL(cudaMemcpyAsync(output.data_ptr(),
                                midway_tensor->ptr(),
                                midway_tensor->num_bytes(),
                                cudaMemcpyDeviceToDevice,
                                cuda_stream));

      // copy input to gpu tensor
      CUDA_CALL(cudaMemcpyAsync(midway_tensor->ptr(),
                                input.data_ptr(),
                                midway_tensor->num_bytes(),
                                cudaMemcpyDeviceToDevice,
                                cuda_stream));
    }

    // wait memory copy finish
    auto wait_event = dadt::obtain_cuda_event();

    // put wait event into stream and wait event finish
    CUDA_CALL(cudaEventRecord(wait_event, cuda_stream));
    CUDA_CALL(cudaEventSynchronize(wait_event));

    // for now the midway result has been copy to output and input has copy in midway tesnor
    // when copy finish create a task put into task queue to do all reduce
    dadt::Task task;
    task.name = name;
    task.tensor = midway_tensor;
    task.task_type = dadt::kAllReduceTaskType;

    task.done = [midway_tensor] {
      midway_tensor->wait(dadt::LockTensorStatus::kInExecute, dadt::LockTensorStatus::kWaitForFetch);
    };

    // change tensor status
    midway_tensor->wait(dadt::LockTensorStatus::kInFetch, dadt::LockTensorStatus::kInExecute);

    // put task in queue
    dadt::enqueue_task(std::move(task));
  };

  // put the job into threadpool
  dadt::thread_pool_enqueue(std::move(job));
}
#endif

void all_reduce_async(torch::Tensor input, torch::Tensor output, const std::string &name, float multiplier) {
  if (input.is_cuda()) {
#ifdef HAVE_NCCL
    return all_reduce_gpu_async(input, output, name, multiplier);
#else
    RUNTIME_ERROR("dadt not build with GPU, please rebuild it with GPU.");
#endif
  } else {
    // return all_reduce_cpu(input, name, multiplier);
    RUNTIME_ERROR("not write");
  }
}

// wait all reduce finish
void wait_all_reduce_finish() {
  dadt::thread_pool_wait();
}

// Define API in python module.
PYBIND11_MODULE(dadt_pytorch, m) {
  m.def("broad_cast", &broad_cast, "broad_cast tensor from rank-0 to other ranks.");
  m.def("all_reduce", &all_reduce, "all_reduce cross all rank.");
  //m.def("all_reduce_async", &all_reduce_async, "all_reduce cross all rank async, must call sync before use grad.");
  //m.def("wait_all_reduce_finish", &wait_all_reduce_finish, "wait all reduce finish.");
}


}
}


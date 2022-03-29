#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "common/context.h"
#include "pytorch.h"

namespace dadt {
namespace pytorch {

// Define API in python module.
PYBIND11_MODULE(dadt_pytorch, m) {
  pybind11::class_<Config>(m, "Config")
      .def(pybind11::init<>())
      .def_readwrite("cycle_duration_ms", &Config::cycle_duration_ms)
      .def_readwrite("executor_type", &Config::executor_type)
      .def_readwrite("all_reduce_buffer_size", &Config::all_reduce_buffer_size)
      .def_readwrite("group_buffer_size", &Config::group_buffer_size)
      .def_readwrite("gpu_device_id", &Config::gpu_device_id);

  m.def("initialize", &Initialize, pybind11::arg("config"));

  m.def("shutdown", &Shutdown);

  m.def("initialized", &Initialized);

  m.def("size", &Size);

  m.def("local_size", &LocalSize);

  m.def("rank", &Rank);

  m.def("local_rank", &LocalRank);

  m.def("barrier", &Barrier);

  m.def("local_barrier", &LocalBarrier);

  m.def("broad_cast", &BroadCast, pybind11::arg("id"), pybind11::arg("input"));

  m.def("all_reduce", &AllReduce, pybind11::arg("id"), pybind11::arg("input"));

  m.def("all_reduce_async", &AllReduceAsync, pybind11::arg("id"),
        pybind11::arg("input"));

  m.def("coo_all_reduce", &CooAllReduce, pybind11::arg("id"),
        pybind11::arg("input"));

  m.def("coo_all_reduce_async", &CooAllReduceAsync, pybind11::arg("id"),
        pybind11::arg("input"));
}

}  // namespace pytorch
}  // namespace dadt

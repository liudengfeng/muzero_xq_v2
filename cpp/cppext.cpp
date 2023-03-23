#include <torch/extension.h>
#include "config.hpp"
#include "mcts.hpp"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("infer", &infer, "测试推理");
    // m.def("backward", &lltm_backward, "LLTM backward");

    py::class_<MuZeroConfig>(m, "MuZeroConfig")
        .def(py::init<>())
        .def_readwrite("seed", &MuZeroConfig::seed)
        .def_readonly("observation_shape", &MuZeroConfig::observation_shape);
}
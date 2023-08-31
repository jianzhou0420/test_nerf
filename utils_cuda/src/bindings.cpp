#include <torch/extension.h>

#include "utils.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("get_xyz", &get_xyz, "get_xyz (CUDA)");
    m.def("test", &test, "test (CUDA)");
}
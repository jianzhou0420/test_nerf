#include <torch/extension.h>

#include "hash_grid_accelerator.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("grid_encode_forward", &grid_encode_forward, "grid_encode_forward (CUDA)");
    m.def("grid_encode_backward", &grid_encode_backward, "grid_encode_backward (CUDA)");
}




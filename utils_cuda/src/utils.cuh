
#include <stdint.h>
#include <torch/torch.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <ATen/cuda/CUDAContext.h>
#include <torch/torch.h>

#include <algorithm>
#include <stdexcept>

#include <stdint.h>
#include <cstdio>

void get_xyz(const at::Tensor c2w, const at::Tensor direction, const at::Tensor depth, at::Tensor point2world);

__global__ void get_xyz_kernel(const float *c2w, const float *direction, const float *depth, float *point2world, int N);
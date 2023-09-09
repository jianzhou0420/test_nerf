
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

// 1. Interface
void grid_encode_forward(const at::Tensor input, const at::Tensor memory, 
                         const at::Tensor offset, const at::Tensor output,
                         const at::Tensor resolution_list, const at::Tensor side_length_list, 
                         const uint32_t T);

void grid_encode_backward(const at::Tensor grad,
                          const at::Tensor input, const at::Tensor memory, 
                          const at::Tensor offset, const at::Tensor new_memory_head,
                          const at::Tensor resolution_list, const at::Tensor side_length_list,
                          const uint32_t T);

// 2. Wrapper::
template <typename scalar_t>
void grid_forward_wrapper_1(const float *input, const scalar_t *memory, 
                            const int *offset, scalar_t *output,
                            const int *resolution_list, const float *side_length_list,
                            const uint32_t D, const uint32_t F, 
                            const uint32_t L, const uint32_t N, const uint32_t T);

template <typename scalar_t, uint32_t D>
void grid_forward_wrapper_2(const float *input, const scalar_t *memory, 
                            const int *offset, scalar_t *output,
                            const int *resolution_list, const float *side_length_list,
                            const uint32_t F, 
                            const uint32_t L, const uint32_t N, const uint32_t T);

template <typename scalar_t>
void grid_backward_wrapper_1(const scalar_t *grad,
                             const float *input, const int *offset, 
                             scalar_t *new_memory_head,
                             const int *resolution_list, const float *side_length_list,
                             const uint32_t D, const uint32_t F, 
                             const uint32_t L, const uint32_t N, const uint32_t T);

template <typename scalar_t, uint32_t D>
void grid_backward_wrapper_2(const scalar_t *grad,
                             const float *input, const int *offset, 
                             scalar_t *new_memory_head,
                             const int *resolution_list, const float *side_length_list,
                             const uint32_t F, 
                             const uint32_t L, const uint32_t N, const uint32_t T);

// 3. Kernel:
template <typename scalar_t, uint32_t D, uint32_t F>
__global__ void kernel_grid_forward(const float *__restrict__ input, const scalar_t *__restrict__ memory_head, 
                                    const int *__restrict__ offset, scalar_t *__restrict__ output,
                                    const int *__restrict__ resolution_list, const float *__restrict__ side_length_list,
                                    const uint32_t L, const uint32_t N, const uint32_t T);

template <typename scalar_t, uint32_t D, uint32_t F>
__global__ void kernel_grid_backward(const scalar_t *__restrict__ grad, 
                                     const float *__restrict__ input, 
                                     const int *__restrict__ offset, scalar_t *__restrict__ new_memory_head,
                                     const int *__restrict__ resolution_list, const float *__restrict__ side_length_list,
                                     const uint32_t L, const uint32_t N, const uint32_t T);

//

// 4. Utils
/**
 *Getting Idx instead of features.
 *Reason: For forward pass, getting features are convinent, but in backward pass, we dont need features, we need idx.
 *also, getting features is quite easy in parent function.
 **/

template <typename scalar_t>
__device__ uint32_t get_features_from_hash_table(const scalar_t *pos_grid, const uint32_t D, const uint32_t T)
{
    // x,y,z are the coordinates of the point
    // this_memory_head is the head of the memory level
    // F is the dimension of the features2 654 435 761, and 𝜋3 = 805 459 861.

    // 在tncc的common_device.h,line 690中，它有stride去规范tiled or hashed，我可以不用它的方法，自己写。

    const uint32_t primes[3] = {1, 2654435761, 805459861};
    uint32_t middle = 0;
    uint32_t this_idx = 0;
    int F = 28;

#pragma unroll
    for (uint32_t i = 0; i < D; ++i)
    {
        middle ^= pos_grid[i] * primes[i];
    }
    this_idx = middle % T *F;

    return this_idx;
};


__device__ uint32_t get_features_from_tile(const uint32_t *pos_grid, const uint32_t F, const int *resolution_list)
{

    // tile的话，直接计算head在哪就行了
    uint32_t this_idx = 0;


    this_idx = (pos_grid[0]*(resolution_list[1]*resolution_list[2]) +pos_grid[1]*resolution_list[2]+ pos_grid[2])*F; //TODO: 
    return this_idx;
};

// just for compatability of half precision in AT_DISPATCH_FLOATING_TYPES_AND_HALF... program will never reach here!
__device__ inline at::Half atomicAdd(at::Half *address, at::Half val)
{
    // requires CUDA >= 10 and ARCH >= 70
    // this is very slow compared to float or __half2, never use it.
    // return atomicAdd(reinterpret_cast<__half*>(address), val);
}

template <typename T>
__device__ void printTemplateVariable(const T &variable)
{
    if constexpr (sizeof(T) == sizeof(float))
    {
        printf("float value: %f\n", variable);
    }
    else if constexpr (sizeof(T) == sizeof(double))
    {
        printf("Double value: %f\n", variable);
    }
    else if constexpr (sizeof(T) == sizeof(const char *))
    {
        printf("String value: %s\n", variable);
    }
    else
    {
        printf("Unknown type\n");
    }
}
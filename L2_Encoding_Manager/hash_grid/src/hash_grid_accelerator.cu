// Declare some coding styles:
// 1. only use pointer to pass array
// 2. return by pointer
// 3. variables consistent in all project like DFLN.
// 4. grouping the parameters into groups
// 5. use singular instead of mixed singular plural to make the variable name more consistent
// 5. use as fewer redirection as possible, except for the case that the redirection is necessary for the code to be more readable.
// 6. use the same naming style as Python guide: https://www.python.org/dev/peps/pep-0008/#naming-conventions
//    This can help to make the code more readable and consistent with the Python code. As people use my code are commonly Python users, this can help them to understand the code more easily.

#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <ATen/cuda/CUDAContext.h>
#include <torch/torch.h>

#include <algorithm>
#include <stdexcept>
#include <cstdint>

#include <stdint.h>
#include <cstdio>
// #include "hash_grid_accelerator.cuh"


#define CHECK_IN_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x, " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x, " must be contiguous")
#define CHECK_IS_INT(x) TORCH_CHECK(x.scalar_type() == at::ScalarType::Int, #x, " must be int")
#define CHECK_IS_FLOAT(x) TORCH_CHECK(x.scalar_type() == at::ScalarType::Float || x.scalar_type() == at::ScalarType::Half || x.scalar_type() == at::ScalarType::Double, #x " must be a floating tensor")

#include "hash_grid_accelerator.cuh"
#include <math.h>

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// Function parameters explanation and grouping:

// Group1: input, memory, offset, output
// input (float): Tensor[num_pixels,D],
// memory (uint_32): Tensor[grid_size, F]  grid_size is defined by offset
// offset (int): Tensor[L]  In shape of [L], each element is the size of each level
// output (uint_32): Tensor[num_pixels, L, F] the output of the encoder (remember, we don't apply MLP at this stage)

// Group2: resolution_list, side_length_list
// resolution_list (int): Tensor[L]  In shape of [L], each element is the resolution of each level
// side_length_list (float): Tensor[L]  In shape of [L], each element is the side length of each level

// Group3: D, F, L, N
// D(int): dimension of the input
// F(int): encoding features, the number is 2 in the paper
// L(int): number of hash levels
// N(int): number of pixels
// T(int): number of max entries

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// Data Type Explanation:


///////////////////////////////////////////////////////////////////////////
//////////////////////////Section 1: forward part//////////////////////////
///////////////////////////////////////////////////////////////////////////
void grid_encode_forward(const at::Tensor input, const at::Tensor memory, const at::Tensor offset, const at::Tensor output,
                         const at::Tensor resolution_list, const at::Tensor side_length_list, const uint32_t T)
{
    // input are the coordinates of the points
    // memory is the grid container
    // offset records how to divide the grid
    // output is the rgb value of the points
    // 我觉得他们不应该是const。
    CHECK_IN_CUDA(input);
    CHECK_IN_CUDA(memory);
    CHECK_IN_CUDA(offset);
    CHECK_IN_CUDA(output);

    CHECK_CONTIGUOUS(input);
    CHECK_CONTIGUOUS(memory);
    CHECK_CONTIGUOUS(offset);
    CHECK_CONTIGUOUS(output);

    CHECK_IS_FLOAT(input);
    CHECK_IS_FLOAT(memory);
    CHECK_IS_INT(offset);
    CHECK_IS_FLOAT(output);

    // to make it more clear, we use some variables to represent the shape of the tensors instead of using size directly
    const uint32_t D = input.size(1);
    const uint32_t F = memory.size(1);
    const uint32_t L = offset.size(0);
    const uint32_t N = input.size(0);
    printf("checkpoint1_grid_encode_forward");
    // T is the number of last offset

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(memory.scalar_type(), "grid_forward",
                                        ([&]
                                         { grid_forward_wrapper_1<scalar_t>(input.data_ptr<float>(), memory.data_ptr<scalar_t>(), offset.data_ptr<int>(), output.data_ptr<scalar_t>(),
                                                                                   resolution_list.data_ptr<int>(), side_length_list.data_ptr<float>(),
                                                                                   D, F, L, N, T); }));

    // TODO: support AI_DISPATCH_FLOATING_TYPES_AND_HALF
}

template<typename scalar_t>
void grid_forward_wrapper_1(const float *input, const scalar_t *memory, const int *offset, scalar_t *output,
                            const int *resolution_list, const float *side_length_list,
                            const uint32_t D, const uint32_t F, const uint32_t L, const uint32_t N, const uint32_t T)
{
    //test
    printf("checkpoint2_grid_forward_wrapper_1");
    // test
    switch (D)
    {
    case 1:
        grid_forward_wrapper_2<scalar_t,1>(input, memory, offset, output, resolution_list, side_length_list, F, L, N, T);
        break;
    case 2:
        grid_forward_wrapper_2<scalar_t,2>(input, memory, offset, output, resolution_list, side_length_list, F, L, N, T);
        break;
    case 3:
        grid_forward_wrapper_2<scalar_t,3>(input, memory, offset, output, resolution_list, side_length_list, F, L, N, T);
        break;
    default:
        break;
    }
    //
}

template <typename scalar_t, uint32_t D>
void grid_forward_wrapper_2(const float *input, const scalar_t *memory, const int *offset, scalar_t *output,
                            const int *resolution_list, const float *side_length_list,
                            const uint32_t F, const uint32_t L, const uint32_t N, const uint32_t T)
{   // test
    printf("checkpoint3_grid_forward_wrapper_2");
    // /test
    static constexpr const uint32_t num_max_threads_per_block = 1024;     // it is constrained by the GPU type, //TODO: create a function to calculate the number base on the GPU properties
    const uint32_t num_threads_divisions = N / num_max_threads_per_block + 1; // 这里也许有数值问题，+1为了保证富余。超出部分会在kernel中停止计算，所以不用担心。
    const dim3 num_blocks = {(uint32_t)num_threads_divisions, (uint32_t)L, 1};
    switch (F)
    {
    case 1:
        kernel_grid_forward<scalar_t,D, 1><<<num_blocks, num_max_threads_per_block>>>(input, memory, offset, output,
                                                                             resolution_list, side_length_list,
                                                                             L, N, T);
        break; 
    case 2:
        kernel_grid_forward<scalar_t,D, 2><<<num_blocks, num_max_threads_per_block>>>(input, memory, offset, output,
                                                                             resolution_list, side_length_list,
                                                                             L, N, T); 
        break;
    case 3:
        kernel_grid_forward<scalar_t,D, 3><<<num_blocks, num_max_threads_per_block>>>(input, memory, offset, output,
                                                                             resolution_list, side_length_list,
                                                                             L, N, T); 
        break;
    default:
        break;
    }
}

template <typename scalar_t, uint32_t D, uint32_t F>
__global__ void kernel_grid_forward(const float *__restrict__ input, const scalar_t *__restrict__ memory_head, const int *__restrict__ offset, scalar_t *__restrict__ output,
                                    const int *__restrict__ resolution_list, const float *__restrict__ side_length_list,
                                    const uint32_t L, const uint32_t N, const uint32_t T)
{ // to minimize the computation, get more input{
    uint32_t batchID = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t levelID = blockIdx.y;
    printf("checkpoint4_kernel_grid_forward");
    // 保证不超出
    if (batchID >= N) return;

    // Step 1/5: locate, and define local containers

    // test

    memory_head += offset[levelID] * F; // level head
    input += batchID * D;
    output += batchID * levelID * F;

    // std::copy(input,input+D,pos); std is not commonly allowed in cuda runtime, as they have different memory allocation methods
    // instead, use the followings:

    // Step 2/5: Query: get the features of the voxel corners
    // first judge whether to tile or to hash

    // float result_features[F];
    float pos[D];
    float pos_idx[D];
    float pos_min_idx[D]; // the down left corner of the voxel
    float pos_max_idx[D]; // the up right corner of the voxel

    // uint32_t corner_idx[8][3];
    uint32_t features_result[F];
    uint32_t frac[D];

// get the index of Point.
#pragma unroll
    for (int i = 0; i < D; i++)
    {
        pos[i] = input[i];
        pos_idx[i] = pos[i] / side_length_list[levelID]; // the pos_idx is float type. to indicate the position of the voxel it belongs to.
        pos_min_idx[i] = floor(pos_idx[i]);
        pos_max_idx[i] = ceil(pos_idx[i]);
        frac[i] = pos_idx[i] - pos_min_idx[i];
    }

    const uint32_t hashmap_size = offset[levelID + 1] - offset[levelID]; // how many entries in this level
    bool is_hash = false;
    if (hashmap_size >= 524288)
        is_hash = true; // 524288 is the maximum number of entries in a hashmap
    else
        is_hash = false;

// interpolation
#pragma unroll
    for (uint32_t idx = 0; idx < (1 << D); idx++)
    { // 循环2^D次，每次循环计算一个corner的feature
        float w = 1;
        uint32_t pos_grid_local[D];

#pragma unroll
        for (uint32_t d = 0; d < D; d++)
        {
            // 循环对于每个维度循环一次，对应c000 = (1 - x_frac) * (1 - y_frac) * (1 - z_frac)这个
            // 那么如何知道此时是（1-x_frac） 还是（frac）呢？ 因为xyz每个位置要么是0要么是1，所以可以用位运算即可。3个位置，8个可能即为2^3。
            // idx是1-8的数字。根据idx的二进制表示，可以知道xyz的位置对应的是0还是1，且循环不重复。
            // 因此，如下判断就是用来idx这个三位二进制数的每一位是否为0，如果为0，那么就是1-frac，如果为1，那么就是frac
            if ((idx & (1 << d)) == 0)
            {
                w *= 1 - frac[d];
                pos_grid_local[d] = pos_min_idx[d]; // 一个是1-x,一个是x，因为越近数值越小，但权重越大.
            }
            else
            {
                w *= frac[d];
                pos_grid_local[d] = pos_max_idx[d];
            }
        }

        uint32_t location;
        // hash 与否只决定了如何提取features
        if (is_hash)
        {

            location = get_features_from_hash_table(pos_grid_local, D, F);
        }
        else
        {

            location = get_features_from_tile(pos_grid_local, F);
        }

// writing to register (fast)
#pragma unroll
        for (uint32_t ch = 0; ch < F; ch++)
        {
            features_result[ch] += w * memory_head[location + ch];
        }

        // printf("[b=%d, l=%d] uint32_t %d, idx %d, w %f, val %f\n", b, level, idx, index, w, grid[index]);
    }

    // from now, I have the features in this level, I need to write it to the output

#pragma unroll
    for (uint32_t ch = 0; ch < F; ch++)
    {
        output[ch] = features_result[ch];
    }
    printf("checkpoint5_end");
}

//////////////////////////////////////////////////////////////////////////////
////////////////////////// Section 2: Backward pass //////////////////////////
//////////////////////////////////////////////////////////////////////////////



/**
 * The main purpose of each kernel is to allocate the gradient to the corresponding voxel
 * For each pixel in a batch, we have the gradient of each feature in each level of them, we need to find who to blame for the gradient.
 *
 * (Allocating gradients only require to know the weights of interpolation. Calculating the weights don't need to know the features in that vertex. )
 */

void grid_encode_backward(const at::Tensor grad,
                          const at::Tensor input, const at::Tensor memory, const at::Tensor offset, const at::Tensor new_memory_head,
                          const at::Tensor resolution_list, const at::Tensor side_length_list,
                          const uint32_t T)
{
    CHECK_IN_CUDA(grad);
    CHECK_IN_CUDA(input);
    CHECK_IN_CUDA(memory);
    CHECK_IN_CUDA(offset);
    CHECK_IN_CUDA(new_memory_head);

    CHECK_CONTIGUOUS(grad);
    CHECK_CONTIGUOUS(input);
    CHECK_CONTIGUOUS(memory);
    CHECK_CONTIGUOUS(offset);
    CHECK_CONTIGUOUS(new_memory_head);

    CHECK_IS_FLOAT(grad);
    CHECK_IS_FLOAT(input);
    CHECK_IS_FLOAT(memory);
    CHECK_IS_INT(offset);
    CHECK_IS_FLOAT(new_memory_head);

    const uint32_t D = input.size(1);
    const uint32_t F = memory.size(1);
    const uint32_t L = offset.size(0) - 1; // # the first element is 0, the last is the total number of entries,middle ones are the head of each level, so we need to minus 1 to get Level info
    const uint32_t N = input.size(0);

    // TODO: change the name of new_memory_head
    // TODO: AT_DISPATCH_FLOATING_TYPES
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(grad.scalar_type(), "grid_backward",
                                        ([&]
                                         { grid_backward_wrapper_1<scalar_t>(grad.data_ptr<scalar_t>(),
                                                                             input.data_ptr<float>(), offset.data_ptr<int>(), new_memory_head.data_ptr<scalar_t>(),
                                                                             resolution_list.data_ptr<int>(), side_length_list.data_ptr<float>(),
                                                                             D, F, L, N, T); }));
    // grid_backward_wrapper_1(grad.data_ptr<uint32_t>(),
    //                         input.data_ptr<float>(), offset.data_ptr<uint32_t>(), new_memory_head.data_ptr<uint32_t>(),
    //                         resolution_list.data_ptr<uint32_t>(), side_length_list.data_ptr<float>(),
    //                         D, F, L, N, T);
}

template<typename scalar_t>
void grid_backward_wrapper_1(const scalar_t *grad,
                             const float *input, const int *offset, scalar_t *new_memory_head,
                             const int *resolution_list, const float *side_length_list,
                             const uint32_t D, const uint32_t F, const uint32_t L, const uint32_t N, const uint32_t T)
{
    switch (D)
    {
    case 1:
        grid_backward_wrapper_2<scalar_t,1>(grad,
                                   input, offset, new_memory_head,
                                   resolution_list, side_length_list,
                                   F, L, N, T);
        break;
    case 2:
        grid_backward_wrapper_2<scalar_t,1>(grad,
                                   input, offset, new_memory_head,
                                   resolution_list, side_length_list,
                                   F, L, N, T);
        break;
    case 3:
        grid_backward_wrapper_2<scalar_t,1>(grad,
                                   input, offset, new_memory_head,
                                   resolution_list, side_length_list,
                                   F, L, N, T);
        break;
    default:
        break;
    }
}

template <typename scalar_t,uint32_t D>
void grid_backward_wrapper_2(const scalar_t *grad,
                             const float *input, const int *offset, scalar_t *new_memory_head,
                             const int *resolution_list, const float *side_length_list,
                             const uint32_t F, const uint32_t L, const uint32_t N, const uint32_t T)
{

    static constexpr const uint32_t num_max_threads_per_block = 1024;         // it is constrained by the GPU type, //TODO: create a function to calculate the number base on the GPU properties
    const uint32_t num_threads_divisions = N / num_max_threads_per_block + 1; // 这里也许有数值问题，+1为了保证富余。超出部分会在kernel中停止计算，所以不用担心。
    const dim3 num_blocks = {(uint32_t)num_threads_divisions, (uint32_t)L, 1};

    switch (D)
    {
    case 1:
        kernel_grid_backward<scalar_t,D, 1><<<num_blocks, num_max_threads_per_block>>>(grad,
                                                                              input, offset, new_memory_head,
                                                                              resolution_list, side_length_list,
                                                                              L, N, T);
        break;
    case 2:
        kernel_grid_backward<scalar_t,D, 2><<<num_blocks, num_max_threads_per_block>>>(grad,
                                                                              input, offset, new_memory_head,
                                                                              resolution_list, side_length_list,
                                                                              L, N, T);
    case 3:
        kernel_grid_backward<scalar_t,D, 3><<<num_blocks, num_max_threads_per_block>>>(grad,
                                                                              input, offset, new_memory_head,
                                                                              resolution_list, side_length_list,
                                                                              L, N, T);
    default:
        break;
    }
}

template <typename scalar_t, uint32_t D, uint32_t F>
__global__ void kernel_grid_backward(const scalar_t *__restrict__ grad, // grad has shape [L,N,F] the grad in each feature in each pixel in each level
                                     const float *__restrict__ input, const int *__restrict__ offset, scalar_t *__restrict__ new_memory_head,
                                     const int *__restrict__ resolution_list, const float *__restrict__ side_length_list,
                                     const uint32_t L, const uint32_t N, const uint32_t T)
{
    //

    const uint32_t batchID = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t levelID = blockIdx.y;
    // 保证不超出
    if (batchID >= N)
        return;

    grad += (levelID * N + batchID) * F;
    input += batchID * D;
    new_memory_head += offset[levelID] * F; // Actually is the output, it is got from grid_embeddings=torch.zeros_like(embeddings)

    // Step 2/5: Query: get the features of the voxel corners
    // first judge whether to tile or to hash

    // float result_features[F];
    // Step 1/5: locate, and define local containers
    float pos[D];
    float pos_idx[D];     // It indeed is a float. we use the float to find the upper and lower idx of the voxel it belongs to.
    float pos_min_idx[D]; // the down left corner of the voxel
    float pos_max_idx[D]; // the up right corner of the voxel

    // uint32_t corner_idx[8][3];
    // uint32_t features_result[F];
    uint32_t frac[D];

// get the index of Point.
#pragma unroll
    for (int i = 0; i < D; i++)
    {
        pos[i] = input[i];
        pos_idx[i] = pos[i] / side_length_list[levelID]; // the pos_idx is float type. to indicate the position of the voxel it belongs to.
        pos_min_idx[i] = floor(pos_idx[i]);
        pos_max_idx[i] = ceil(pos_idx[i]);
        frac[i] = pos_idx[i] - pos_min_idx[i];
    }

    const uint32_t hashmap_size = offset[levelID + 1] - offset[levelID]; // how many entries in this level
    bool is_hash = false;
    if (hashmap_size >= 524288)
        is_hash = true; // 524288 is the maximum number of entries in a hashmap
    else
        is_hash = false;

    uint32_t grad_cur[D] = {0}; // fetch to register
#pragma unroll
    for (uint32_t c = 0; c < D; c++)
    {
        grad_cur[c] = grad[c];
    }

// interpolation //TODO: options to sacrify the memory to speed up, as the have already calculated the weights in the forward pass.
#pragma unroll
    for (uint32_t idx = 0; idx < (1 << D); idx++)
    { // 循环2^D次，每次循环计算一个corner的feature
        float w = 1;
        uint32_t pos_grid_local[D];

#pragma unroll
        for (uint32_t d = 0; d < D; d++)
        {
            // 循环对于每个维度循环一次，对应c000 = (1 - x_frac) * (1 - y_frac) * (1 - z_frac)这个
            // 那么如何知道此时是（1-x_frac） 还是（frac）呢？ 因为xyz每个位置要么是0要么是1，所以可以用位运算即可。3个位置，8个可能即为2^3。
            // idx是1-8的数字。根据idx的二进制表示，可以知道xyz的位置对应的是0还是1，且循环不重复。
            // 因此，如下判断就是用来idx这个三位二进制数的每一位是否为0，如果为0，那么就是1-frac，如果为1，那么就是frac
            if ((idx & (1 << d)) == 0)
            {
                w *= 1 - frac[d];
                pos_grid_local[d] = pos_min_idx[d]; // 一个是1-x,一个是x，因为越近数值越小，但权重越大.
            }
            else
            {
                w *= frac[d];
                pos_grid_local[d] = pos_max_idx[d];
            }
        }

        uint32_t location = 0;
        // hash 与否只决定了如何提取features
        if (is_hash)
        {

            location = get_features_from_hash_table(pos_grid_local, D, F);
        }
        else
        {

            location = get_features_from_tile(pos_grid_local, F);
        }

        // atomicAdd for __half is slow (especially for large values), so we use __half2 if N_C % 2 == 0
        // TODO: use float which is better than __half, if N_C % 2 != 0
        if (std::is_same<scalar_t, at::Half>::value && D % 2 == 0)
        {
            #pragma unroll
            for (uint32_t c = 0; c < D; c += 2)
            {
                // process two __half at once (by interpreting as a __half2)
                __half2 v = {(__half)(w * grad_cur[c]), (__half)(w * grad_cur[c + 1])};
                atomicAdd((__half2 *)&new_memory_head[location + c], v);
            }
            // float, or __half when N_C % 2 != 0 (which means C == 1)
        }
        else
        {
            #pragma unroll
            for (uint32_t c = 0; c < D; c++)
            {
                atomicAdd(&new_memory_head[location + c], w * grad_cur[c]);
            }
        }

        // printf("[b=%d, l=%d] uint32_t %d, idx %d, w %f, val %f\n", b, level, idx, index, w, grid[index]);
    }
}




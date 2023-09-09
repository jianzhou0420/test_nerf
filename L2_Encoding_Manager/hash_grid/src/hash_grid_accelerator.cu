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
    const uint32_t L = offset.size(0)-1;
    const uint32_t N = input.size(0);

    // T is the number of last offset

    // test
    // printf("memory %d %d\n", memory.size(0), memory.size(1));
    // printf("memory shape %d\n", memory.dim());
    // /test


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

    // /test
    static constexpr const uint32_t num_max_threads_per_block = 1024; // it is constrained by the GPU type, 
    //TODO: create a function to calculate the number base on the GPU properties
    const uint32_t num_threads_divisions = N / num_max_threads_per_block + 1; 
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
    case 28: // test 28
        kernel_grid_forward<scalar_t,D, 28><<<num_blocks, num_max_threads_per_block>>>(input, memory, offset, output,
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
    // input: [N,D]
    // memory_head: [offset[-1],F]
    // offset: [L+1]
    // output: [N,L,F]
    
    uint32_t batchID = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t levelID = blockIdx.y;
  
    
    if (batchID >= N) return;

    // Step 1/5: locate, and define local containers



    memory_head += offset[levelID] * F; // level head
    input += batchID * D;
    output += (batchID * L+levelID) * F;
    resolution_list += levelID*3;

    // test

    // ID
    // printf("batchID: %d, levelID: %d\n", batchID, levelID);

    // /test


    // std::copy(input,input+D,pos); std is not commonly allowed in cuda runtime, as they have different memory allocation methods
    // instead, use the followings:

    // Step 2/5: Query: get the features of the voxel corners
    // first judge whether to tile or to hash

    // float result_features[F];
    float pos[D];
    float pos_idx[D]; // yes it is index and it is float. to indicate the scale of the voxel it belongs to.
    uint32_t pos_min_idx[D]; // the down left corner of the voxel
    uint32_t pos_max_idx[D]; // the up right corner of the voxel

    // uint32_t corner_idx[8][3];
    scalar_t features_result[F]={0};
    float frac[D]; // float, the same reason as pos_idx

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
    // test
    //print all the pos
    // printf("pos: %f %f %f\n", pos[0], pos[1], pos[2]);
    // printf("pos_idx: %f %f %f\n", pos_idx[0], pos_idx[1], pos_idx[2]);
    // printf("pos_min_idx: %d %d %d\n", pos_min_idx[0], pos_min_idx[1], pos_min_idx[2]);
    // printf("pos_max_idx: %d %d %d\n", pos_max_idx[0], pos_max_idx[1], pos_max_idx[2]);
    // printf("frac: %f %f %f\n", frac[0], frac[1], frac[2]);

    // printf("side_length_list[levelID]: %f\n", side_length_list[levelID]);
    // /test

    const uint32_t hashmap_size = offset[levelID + 1] - offset[levelID]; // how many entries in this level
    bool is_hash = false;
    if (hashmap_size >= 524288)
        is_hash = true; // 524288 is the maximum number of entries in a hashmap
    else
        is_hash = false;

// interpolation
#pragma unroll
    for (uint32_t idx = 0; idx < (1 << D); idx++)
    {  //iterates over a range of values from 0 up to (2^D) - 1, each time calculate one vertex
        float w=1; // for each vertex, it has separate weights for each dimension
        uint32_t pos_grid_local[D];

#pragma unroll
        for (uint32_t d = 0; d < D; d++) // this loop process each dimension
        {
            // Initialize each weight to 1
            // Loop for each dimension corresponding to c000 = (1 - x_frac) * (1 - y_frac) * (1 - z_frac)
            // How do we determine if it's (1 - x_frac) or (x_frac)? 
            // Since the position for xyz can either be 0 or 1, bit manipulation can be employed. 
            // With three positions, we have 8 possibilities, i.e., 2^3.
            // 'idx' is a number ranging from 0 to 7. 
            // Based on the binary representation of 'idx', 
            // we can ascertain if the position of xyz corresponds to 0 or 1 without any repetition.
            // Thus, the following checks each bit of the three-bit binary number represented by 'idx'. 
            // If a bit is 0, it corresponds to (1 - frac); if it's 1, it corresponds to frac.
            if ((idx & (1 << d)) == 0)
            {   
                // /test
                w *= 1 - frac[d];
                pos_grid_local[d] = pos_min_idx[d];
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
            location = get_features_from_tile(pos_grid_local, F, resolution_list);
        }


        // test

      
  
        // /test


// writing to register (fast)
#pragma unroll
        for (uint32_t ch = 0; ch < F; ch++) //each feature in one vertex share the same weight
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

        // test
        // printTemplateVariable(output[levelID*F+ch]);
        // printTemplateVariable(doubleValue);
        // printTemplateVariable(stringValue);
        // printf("BatchID: %d, LevelID: %d, output[%d]=%f", batchID,levelID,ch, output[ch];
        // printf("features_result[%d]=%f\n",ch,features_result[ch]);
        // /test
    }
    // ok,Now except the hash part, the forward pass is valid
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
        grid_backward_wrapper_2<scalar_t,2>(grad,
                                   input, offset, new_memory_head,
                                   resolution_list, side_length_list,
                                   F, L, N, T);
        break;
    case 3:
        grid_backward_wrapper_2<scalar_t,3>(grad,
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

    switch (F)
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
        break;
    case 28:
        kernel_grid_backward<scalar_t,D, 28><<<num_blocks, num_max_threads_per_block>>>(grad,
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
    // assume we have total grad and we want to allocate it to each voxel, we can simply use the interpolation weights to do this.
    // weight_for_each_voxel = weight_for_each_vertex * total_grad
    // the weight for each voxel has been calculated in forward pass
    // However, storing that weight means we need another [N,L,8]*float32 size momory.
    // You can definitely scarify some memory to get a faster speed. wait. Is it realy wasting memoery? I don't know. that needs verriation.


    // TODO: change the new_memory_head to grad_memory_head
    // grad: [N,L*F] for instance, [816000,448] N=816000,L=16,F=28
    // new_memory_head: [offser[-1],F]

    // the features of one specific vertex share the same weight. so our kernel is invoked according to input, but not these features
    // the logic of the function is as follows:
    // 1. For each input and each level, calculate the 8 surrounding vertices
    // 2. When iterting the 8 vertices, in each loop, calculate the weight of this specific vertex 
    //    and atomicly add the 28 features's grad to the corresponding feature in the vertex. offcouse, we will divide the weight by the weight



    uint32_t batchID = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t levelID = blockIdx.y;

    
    

    // /test
    // 保证不超出
    if (batchID >= N)
        return;

    // Step 1/5: locate, and define local containers

    // memory_head += offset[levelID] * F; // level head
    input += batchID * D;
    new_memory_head += offset[levelID]*F;
    grad+=batchID*L*F+levelID*F;
    resolution_list+=levelID*3;

    // printf("levelID%d \n",levelID);
    // printf("offset%d \n",offset[levelID]);
    // printf("F%d \n",F);

    // test

    // ID
    // printf("batchID: %d, levelID: %d\n", batchID, levelID);

    // /test

    // std::copy(input,input+D,pos); std is not commonly allowed in cuda runtime, as they have different memory allocation methods
    // instead, use the followings:

    // Step 2/5: Query: get the features of the voxel corners
    // first judge whether to tile or to hash

    // float result_features[F];
    float pos[D];
    float pos_idx[D];        // yes it is index and it is float. to indicate the scale of the voxel it belongs to.
    uint32_t pos_min_idx[D]; // the down left corner of the voxel
    uint32_t pos_max_idx[D]; // the up right corner of the voxel

    // uint32_t corner_idx[8][3];
    float frac[D]; // float, the same reason as pos_idx

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
    // test
    // print all the pos
    // printf("pos: %f %f %f\n", pos[0], pos[1], pos[2]);
    // printf("pos_idx: %f %f %f\n", pos_idx[0], pos_idx[1], pos_idx[2]);
    // printf("pos_min_idx: %d %d %d\n", pos_min_idx[0], pos_min_idx[1], pos_min_idx[2]);
    // printf("pos_max_idx: %d %d %d\n", pos_max_idx[0], pos_max_idx[1], pos_max_idx[2]);
    // printf("frac: %f %f %f\n", frac[0], frac[1], frac[2]);

    // printf("side_length_list[levelID]: %f\n", side_length_list[levelID]);
    // /test

    const uint32_t hashmap_size = offset[levelID + 1] - offset[levelID]; // how many entries in this level
    bool is_hash = false;
    if (hashmap_size >= 524288)
        is_hash = true; // 524288 is the maximum number of entries in a hashmap
    else
        is_hash = false;

// interpolation
#pragma unroll
    for (uint32_t idx = 0; idx < (1 << D); idx++)
    {                // iterates over a range of values from 0 up to (2^D) - 1, each time calculate one vertex
        float w = 1; // for each vertex, it has separate weights for each dimension
        uint32_t pos_grid_local[D];
        // test
        // printf("hello1\n");
        // /test

#pragma unroll
            for (uint32_t d = 0; d < D; d++) // this loop process each dimension
        {
            // test
            // printf("D:%d", D);
            // printf("hello\n");
            // /test
            // initialize each weight to 1
            // 循环对于每个维度循环一次，对应c000 = (1 - x_frac) * (1 - y_frac) * (1 - z_frac)这个
            // 那么如何知道此时是（1-x_frac） 还是（frac）呢？ 因为xyz每个位置要么是0要么是1，所以可以用位运算即可。3个位置，8个可能即为2^3。
            // idx是0-7的数字。根据idx的二进制表示，可以知道xyz的位置对应的是0还是1，且循环不重复。
            // 因此，如下判断就是用来idx这个三位二进制数的每一位是否为0，如果为0，那么就是1-frac，如果为1，那么就是frac
            if ((idx & (1 << d)) == 0)
            {
                
                // /test
                w *= 1 - frac[d];
                pos_grid_local[d] = pos_min_idx[d]; // 一个是1-x,一个是x，因为越近数值越小，但权重越大.
            }
            else
            {
                w *= frac[d];
                pos_grid_local[d] = pos_max_idx[d];
            }
        }

        // For here, you have had the weight for a specific vertex.

        // Locate the memory adress of the vertex 
        uint32_t location;
        if (is_hash)
        {
            location = get_features_from_hash_table(pos_grid_local, D, F);
        }
        else
        {
            location = get_features_from_tile(pos_grid_local, F, resolution_list);
        }

        // Now, you have w:weight & location:memory address

        for (uint32_t ch = 0; ch < F;ch++){

            atomicAdd(&new_memory_head[location + ch], w * grad[ch]);
        }
    }

        // test

        // /test


}




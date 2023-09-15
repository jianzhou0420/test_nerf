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


// #define CHECK_IN_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x, " must be a CUDA tensor")
// #define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x, " must be contiguous")
// #define CHECK_IS_INT(x) TORCH_CHECK(x.scalar_type() == at::ScalarType::Int, #x, " must be int")
// #define CHECK_IS_FLOAT(x) TORCH_CHECK(x.scalar_type() == at::ScalarType::Float || x.scalar_type() == at::ScalarType::Half || x.scalar_type() == at::ScalarType::Double, #x " must be a floating tensor")

// #include "utils.cuh"
#include <math.h>

__global__ void get_xyz_kernel(const float *c2w, const float *direction, const float *depth, float *point2world, const float *bound,int N)
{
    int matrix_dim = 4;
    int batchID = blockDim.x * blockIdx.x + threadIdx.x; // blockDim.x=1024, blockIdx.x=0,1,2,...,N/1024
    int row = threadIdx.y;

    // locate
    if (batchID >= N) return;
    if (row >= matrix_dim-1) return; // mind the row starts from 0



    direction += batchID * 3;
    depth += batchID;
    point2world += batchID * 3;

    float camera_local_point[4] = {0, 0, 0, 1};

#pragma unroll
    for (int i = 0; i < 3; i++)
    {
        camera_local_point[i] = direction[i] * depth[0];
    }


    float sum = 0;

#pragma unroll
    for (int i = 0; i < matrix_dim; i++) 

    {
        sum += c2w[row * matrix_dim + i] * camera_local_point[i];
    }

    point2world[row] = sum - bound[row];
}



void checkCudaInfo(){
    int device;
    cudaGetDevice(&device);

    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, device);

    std::cout << "Device: " << props.name << std::endl;
    std::cout << "Max threads per block: " << props.maxThreadsPerBlock << std::endl;
    std::cout << "Max threads per multiprocessor: " << props.maxThreadsPerMultiProcessor << std::endl;
    std::cout << "Max threads total: " << props.maxThreadsPerMultiProcessor * props.multiProcessorCount << std::endl;
    std::cout << "Max blocks per grid: " << props.maxGridSize[0] << std::endl;
}

void get_xyz(const at::Tensor c2w, const at::Tensor direction, const at::Tensor depth, at::Tensor point2world,at::Tensor bound)
{ // TODO: Shape check  
    // int device;
    // cudaGetDevice(&device);

    // cudaDeviceProp props;
    // cudaGetDeviceProperties(&props, device);

    int N=direction.size(0);
    int threadsx=1024/4;

    const dim3 threads_per_block(threadsx,4);         // it is constrained by the GPU type, //TODO: create a function to calculate the number base on the GPU properties
    const int num_threads_divisions = N / threads_per_block.x + 1; // 这里也许有数值问题，+1为了保证富余。超出部分会在kernel中停止计算，所以不用担心。
    const dim3  blocks_per_grid(num_threads_divisions, 1);

    // print c2w

    get_xyz_kernel<<<blocks_per_grid, threads_per_block>>>(c2w.data_ptr<float>(), direction.data_ptr<float>(), depth.data_ptr<float>(), point2world.data_ptr<float>(),bound.data_ptr<float>(),N);

    // cudaDeviceSynchronize();

}

__global__ void test_kernel()
{
    printf("test_kernel\n");

    return;
}

void test()
{
    printf("test\n");
    test_kernel<<<100, 100>>>();
    return;
}


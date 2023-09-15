#include <stdint.h>

#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <ATen/cuda/CUDAContext.h>
#include <torch/torch.h>

#include <algorithm>
#include <stdexcept>

#include <cstdio>


#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be a contiguous tensor")
#define CHECK_IS_INT(x) TORCH_CHECK(x.scalar_type() == at::ScalarType::Int, #x " must be an int tensor")
#define CHECK_IS_FLOATING(x) TORCH_CHECK(x.scalar_type() == at::ScalarType::Float || x.scalar_type() == at::ScalarType::Half || x.scalar_type() == at::ScalarType::Double, #x " must be a floating tensor")


template <typename T>
__host__ __device__ T div_round_up(T val, T divisor) {
	return (val + divisor - 1) / divisor;
}

template <typename scalar_t>
__global__ void kernel_sh(
    const scalar_t * __restrict__ inputs, 
    scalar_t * outputs, 
    uint32_t B, uint32_t D, uint32_t C,
    scalar_t * dy_dx
) {
	const uint32_t b = threadIdx.x + blockIdx.x * blockDim.x;
	if (b >= B) return;

	const uint32_t C2 = C * C;

	// locate
	inputs += b * D;
	outputs += b * C2;

	scalar_t x = inputs[0], y = inputs[1], z = inputs[2];

	scalar_t xy=x*y, xz=x*z, yz=y*z, x2=x*x, y2=y*y, z2=z*z, xyz=xy*z;
	outputs[0] = 0.28209479177387814f ;                          // 1/(2*sqrt(pi))
	if (C <= 1) { return; }
	outputs[1] = -0.48860251190291987f*y ;                               // -sqrt(3)*y/(2*sqrt(pi))
	outputs[2] = 0.48860251190291987f*z ;                                // sqrt(3)*z/(2*sqrt(pi))
	outputs[3] = -0.48860251190291987f*x ;                               // -sqrt(3)*x/(2*sqrt(pi))
	if (C <= 2) { return; }
	outputs[4] = 1.0925484305920792f*xy ;                                // sqrt(15)*xy/(2*sqrt(pi))
	outputs[5] = -1.0925484305920792f*yz ;                               // -sqrt(15)*yz/(2*sqrt(pi))
	outputs[6] = 0.94617469575755997f*z2 - 0.31539156525251999f ;                         // sqrt(5)*(3*z2 - 1)/(4*sqrt(pi))
	outputs[7] = -1.0925484305920792f*xz ;                               // -sqrt(15)*xz/(2*sqrt(pi))
	outputs[8] = 0.54627421529603959f*x2 - 0.54627421529603959f*y2 ;                              // sqrt(15)*(x2 - y2)/(4*sqrt(pi))
}


// inputs: [B, D], float, in [0, 1]
// outputs: [B, L * C], float
template <typename scalar_t>
void sh_encode_forward_cuda(const scalar_t *inputs, scalar_t *outputs, const uint32_t B, const uint32_t D, const uint32_t C, scalar_t *dy_dx) {
	static constexpr uint32_t N_THREADS = 256;
	kernel_sh<scalar_t><<<div_round_up(B, N_THREADS), N_THREADS>>>(inputs, outputs, B, D, C, dy_dx);
}



void sh_encode_forward(at::Tensor inputs, at::Tensor outputs, const uint32_t B, const uint32_t D, const uint32_t C, at::optional<at::Tensor> dy_dx) {
    CHECK_CUDA(inputs);
    CHECK_CUDA(outputs);
    // CHECK_CUDA(dy_dx);
    
    CHECK_CONTIGUOUS(inputs);
    CHECK_CONTIGUOUS(outputs);
    // CHECK_CONTIGUOUS(dy_dx);

    CHECK_IS_FLOATING(inputs);
    CHECK_IS_FLOATING(outputs);
    // CHECK_IS_FLOATING(dy_dx);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
    inputs.scalar_type(), "sh_encode_forward_cuda", ([&] {
		sh_encode_forward_cuda<scalar_t>(inputs.data_ptr<scalar_t>(), outputs.data_ptr<scalar_t>(), B, D, C, dy_dx.has_value() ? dy_dx.value().data_ptr<scalar_t>() : nullptr);
    }));	
}


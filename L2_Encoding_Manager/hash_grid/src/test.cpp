// 1. Interface
void grid_encode_forward(const at::Tensor input, const at::Tensor memory, const at::Tensor offset, const at::Tensor output,
                         const at::Tensor resolution_list, const at::Tensor side_length_list, const uint32_t T);

void grid_encode_backward(const at::Tensor grad,
                          const at::Tensor input, const at::Tensor memory, const at::Tensor offset, const at::Tensor new_memory_head,
                          const at::Tensor resolution_list, const at::Tensor side_length_list,
                          const uint32_t T);

// 2. Wrapper::
template <typename scalar_t>
void grid_forward_wrapper_1(const float *input, const scalar_t *memory, const int *offset, scalar_t *output,
                            const int *resolution_list, const float *side_length_list,
                            const uint32_t D, const uint32_t F, const uint32_t L, const uint32_t N, const uint32_t T);

template <typename scalar_t, uint32_t D>
void grid_forward_wrapper_2(const float *input, const scalar_t *memory, const int *offset, scalar_t *output,
                            const int *resolution_list, const float *side_length_list,
                            const uint32_t F, const uint32_t L, const uint32_t N, const uint32_t T);

template <typename scalar_t>
void grid_backward_wrapper_1(const scalar_t *grad,
                             const float *input, const int *offset, scalar_t *new_memory_head,
                             const int *resolution_list, const float *side_length_list,
                             const uint32_t D, const uint32_t F, const uint32_t L, const uint32_t N, const uint32_t T);

template <typename scalar_t, uint32_t D>
void grid_backward_wrapper_2(const scalar_t *grad,
                             const float *input, const int *offset, scalar_t *new_memory_head,
                             const int *resolution_list, const float *side_length_list,
                             const uint32_t F, const uint32_t L, const uint32_t N, const uint32_t T);

// 3. Kernel:
template <typename scalar_t, uint32_t D, uint32_t F>
__global__ void kernel_grid_forward(const float *__restrict__ input, const scalar_t *__restrict__ memory_head, 
                                    const int *__restrict__ offset, scalar_t *__restrict__ output,
                                    const int *__restrict__ resolution_list, const float *__restrict__ side_length_list,
                                    const uint32_t L, const uint32_t N, const uint32_t T);

template <typename scalar_t, uint32_t D, uint32_t F>
__global__ void kernel_grid_backward(const scalar_t *__restrict__ grad, // grad has shape [L,N,F] the grad in each feature in each pixel in each level
                                     const float *__restrict__ input, const int *__restrict__ offset, 
                                     scalar_t *__restrict__ new_memory_head,
                                     const int *__restrict__ resolution_list, const float *__restrict__ side_length_list,
                                     const uint32_t L, const uint32_t N, const uint32_t T);

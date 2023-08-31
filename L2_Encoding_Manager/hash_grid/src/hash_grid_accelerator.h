#ifndef _HASH_ENCODE_H
#define _HASH_ENCODE_H

#include <stdint.h>
#include <torch/torch.h>


void grid_encode_forward(const at::Tensor input, const at::Tensor memory, const at::Tensor offset, const at::Tensor output,
                         const at::Tensor resolution_list, const at::Tensor side_length_list,
                         const uint32_t T); // forward pass wrapper

void grid_encode_backward(const at::Tensor grad,
                          const at::Tensor input, const at::Tensor memory, const at::Tensor offset, const at::Tensor new_memory_head,
                          const at::Tensor resolution_list, const at::Tensor side_length_list,
                          const uint32_t T);

#endif
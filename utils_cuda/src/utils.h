#ifndef _HASH_ENCODE_H
#define _HASH_ENCODE_H

#include <stdint.h>
#include <torch/torch.h>

void get_xyz(const at::Tensor c2w, const at::Tensor direction, const at::Tensor depth, at::Tensor point2world,at::Tensor bound);
void test();
#endif
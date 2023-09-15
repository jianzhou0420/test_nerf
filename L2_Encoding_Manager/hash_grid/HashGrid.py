import torch
import torch.nn as nn
from torch.autograd import Function
from torch.cuda.amp import custom_bwd, custom_fwd
import math
import numpy as np
# test


# /test

import _test as _backend

    



# variables explanation:
# inputs: Tensor[num_pixels,D]
# memory: Tensor[grid_size, F]  grid_size is defined by offsets
# offsets: Tensor[L]  In shape of [L], each element is the size of each level
# outputs: Tensor[num_pixels, L, F] the output of the encoder (remember, we don't apply MLP at this stage)
#
#
# D: dimension of the input
# F: encoding features, the number is 2 in the paper
# L: number of hash levels
#

class _grid_encode(Function):
    @staticmethod
    @custom_fwd
    def forward(ctx,
                inputs: torch.Tensor,memory: torch.Tensor,offsets: torch.Tensor,
                resolution_list: torch.Tensor,side_length_list: torch.Tensor,
                T: int
                ):
        # define some variables
        D = inputs.shape[1]
        F = memory.shape[1]
        L = offsets.shape[0]-1# the first element is 0, the last is the total number of entries,middle ones are the head of each level, so we need to minus 1 to get Level info
        N =inputs.shape[0]
        T=T
        
        # if torch.is_autocast_enabled():
        #     memory=memory.to(torch.half)
        # TODO: align corners
        output = torch.zeros((N,L, F), device=inputs.device, dtype=memory.dtype)
        
        # dy_dx=torch.zeros((N,L, 8), device=inputs.device, dtype=memory.dtype) # each point has interpolation weights for 8 points.
        # the location of each vertex can be calculated by the location of the point and the side length of the voxel

        
        #continuous
        inputs=inputs.contiguous()
        memory=memory.contiguous()
        offsets=offsets.contiguous()
        resolution_list=resolution_list.contiguous()
        side_length_list=side_length_list.contiguous()
        output=output.contiguous()
        # dy_dx=dy_dx.contiguous()
        
        
        _backend.grid_encode_forward(inputs, memory, offsets, output,
                                    resolution_list,side_length_list,
                                    T)
        

        ctx.save_for_backward(inputs, memory, offsets, resolution_list,side_length_list)
        ctx.dims = [D,F,L,N,T]
        
        return output
    
    
    @staticmethod
    #@once_differentiable
    @custom_bwd        
    def backward(ctx, grad):
        # print(grad.shape)
        input, memory, offset, resolution_list,side_length_list= ctx.saved_tensors
        D,F,L,N,T=ctx.dims
        grad=grad.view(N,L,F) # .continuous() is used to make the memory together
        grad_memory = torch.zeros_like(memory).to(memory.device)
        grad_memory=grad_memory.contiguous()
        
        _backend.grid_encode_backward(grad,
                                      input,memory,offset,grad_memory,
                                      resolution_list,side_length_list,
                                      T)
        
        # print(grad_memory.shape)
        # print(grad_memory.max())
        # TODO: dy_dx
        return None,grad_memory,None,None,None,None
        
        

# Use it by calling the apply method:
# xdoctest: +SKIP
grid_encode = _grid_encode.apply


class HashGrid(nn.Module):
    def __init__(self):
        super().__init__()
        # follow the notation in the paper


        # Step 1: define some variables
        # hidden variables
        self.interpolation = True
        self.D = 3
        self.bounds = [ [ -3,3 ],[ -4,2.5 ],[ -2,2.5 ] ]
        self.offsets = []  # placeholder for offsets
        self.resolution_list = []  # placeholder for resolution list
        self.side_length_list = []  # side length of each level's voxels

        # variables in paper
        self.L = 16  # number of levels
        self.F = 28 # number of Features
        self.T = 524288  # max entries (hash table size)
        self.N_min = 16
        self.N_max = 2048
        self.b = math.exp((math.log(self.N_max) - math.log(self.N_min)) / (self.L - 1))

        # Step 2: define the levels and register them
        # let's first define the first level
        self.bounds = np.array(self.bounds)
        abs_bounds = self.bounds[:, 1] - self.bounds[:, 0]
        reference_axis_idx = np.argmax(abs_bounds)  # 用最长的来作为参考轴

        this_offset = 0
        self.offsets.append(0)
        for i in range(self.L):
            reference_axis_resolution = self.N_min * self.b ** i
            this_side_length = abs_bounds[reference_axis_idx] / reference_axis_resolution
            this_resolution = np.ceil(abs_bounds / this_side_length)  # ceil to accommodate all the map
            this_resolution = this_resolution.astype(np.int32)
            this_entries = np.min([self.T, np.prod(this_resolution)])  # limit the max number
            # this_entries = np.floor(this_entries / 8) * 8  # optimize for parallel computing我真的看不懂这个，所以先注释掉
            # 如果它是为了储存的高效的话就不管了，反正就那么点内存消耗，不管它

            this_offset += this_entries  # 这里是计算内存head在哪里的offset
            self.side_length_list.append(this_side_length)
            self.offsets.append(this_offset)
            self.resolution_list.append(this_resolution)

        self.offsets = torch.tensor(np.array(self.offsets), dtype=torch.int32)
        self.resolution_list = torch.tensor(np.array(self.resolution_list), dtype=torch.int32)
        self.side_length_list = torch.tensor(np.array(self.side_length_list), dtype=torch.float32)

        # Step 3: define the memory
        self.memory = nn.Parameter(torch.empty((this_offset, self.F)))  # last offset is the total number of entries
        # /test
        # Step 4: initialize the memory
        self.reset_parameters()

    def reset_parameters(self):
        std = 1e-4
        self.memory.data.uniform_(-std, std)
        # test this test is to make a settled output
        # self.memory.data.fill_(std)
        # /test
    def forward(self, input):
        '''
        # Now batch={dict:3}
        #   'points'=Tensor(680*1200,3)
        #   'rgb'=Tensor(680*1200,3)
        #   'depth'=Tensor(680*1200)
        # 不再有'c2w'和'direction'了，就当我在mian里面把它们拼接起来了.
        不对，不需要batch，作为foward的输入，只需要points就可以了。
        :param points_batch: Tensor(680*1200,3)
        :return:
        '''

        # TODO: map to 0-1
        num_point=list(input.shape[:-1])
        
        # TODO: make the to device more suitable
        
        # input=input.to('cuda')
        # self.memory=self.memory.to('cuda') because memory is already on cuda
        self.offsets=self.offsets.to('cuda')
        self.resolution_list=self.resolution_list.to('cuda')
        self.side_length_list=self.side_length_list.to('cuda')
        
        
        
        
        output=grid_encode(input,self.memory,self.offsets,
                           self.resolution_list,self.side_length_list,
                           self.T)

        
        # TODO: find the most efficient way to store for better query speed
        # now the shape is [816000,32]

        return output



    
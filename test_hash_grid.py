import torch
# from get_encoding import get_encoding
import _test as _backend
import math
import numpy as np
from L1_Data_Manager.dataset import get_dataset
from L2_Encoding_Manager.get_encoding import get_encoding
# inner imports
from L0_Traning_Manager.trainner import Tranner
from utils import load_config
from torch.utils.data import DataLoader
from _mynerf_utils import get_xyz,test

import torch.nn as nn
class Timing:
    def __init__(self, name):
        self.name = name

    def __enter__(self):
        self.start = torch.cuda.Event(enable_timing=True)
        self.end = torch.cuda.Event(enable_timing=True)
        self.start.record()

    def __exit__(self, type, value, traceback):
        self.end.record()
        torch.cuda.synchronize()
        print('Time:',self.name, 'elapsed', self.start.elapsed_time(self.end), 'ms')






N=816000



D = 3
bounds = [ [ -3,3 ],[ -4,2.5 ],[ -2,2.5 ] ]
offsets = []  # placeholder for offsets
resolution_list = []  # placeholder for resolution list
side_length_list = []  # side length of each level's voxels

# variables in paper
L = 2 # number of levels
F = 2 # number of Features
T = 524288  # max entries (hash table size)
N_min = 4
N_max = 8
b = math.exp((math.log(N_max) - math.log(N_min)) / (L - 1))

# Step 2: define the levels and register them
# let's first define the first level
bounds = np.array(bounds)
abs_bounds = bounds[:, 1] - bounds[:, 0]
reference_axis_idx = np.argmax(abs_bounds)  # 用最长的来作为参考轴

this_offset = 0
offsets.append(0)
for i in range(L):
    reference_axis_resolution = N_min * b ** i
    this_side_length = abs_bounds[reference_axis_idx] / reference_axis_resolution
    this_resolution = np.ceil(abs_bounds / this_side_length)  # ceil to accommodate all the map
    this_resolution = this_resolution.astype(np.int32)
    this_entries = np.min([T, np.prod(this_resolution)])  # limit the max number
    # this_entries = np.floor(this_entries / 8) * 8  # optimize for parallel computing我真的看不懂这个，所以先注释掉
    # 如果它是为了储存的高效的话就不管了，反正就那么点内存消耗，不管它

    this_offset += this_entries  # 这里是计算内存head在哪里的offset
    side_length_list.append(this_side_length)
    offsets.append(this_offset)
    resolution_list.append(this_resolution)

offsets = torch.tensor(np.array(offsets), dtype=torch.int32)
resolution_list = torch.tensor(np.array(resolution_list), dtype=torch.int32)
side_length_list = torch.tensor(np.array(side_length_list), dtype=torch.float32)

pass


input=torch.tensor([1,1,1],dtype=torch.float32).reshape((1,3))
grad=torch.ones((1,2*2),dtype=torch.float32)
# grad[0,1]=2
# grad[0,3]=100
memory=torch.zeros((offsets[-1],F),dtype=torch.float32)
new_memory=torch.zeros((offsets[-1],F),dtype=torch.float32)


input=input.to('cuda')
grad=grad.to('cuda')  #[]
memory=memory.to('cuda')
new_memory=new_memory.to('cuda')
offsets=offsets.to('cuda')
resolution_list=resolution_list.to('cuda')
side_length_list=side_length_list.to('cuda')

input=input.contiguous()

for j in range(2):
    _backend.grid_encode_backward(grad,
                                input,memory,offsets,new_memory,
                                resolution_list,side_length_list,
                                T)

for i,line in enumerate(new_memory):
    if line.sum()!=0:
        print(line,i)
# print(new_memory)
# print(new_memory[8:16,:])








#################### test forward pass##########################################

# config = load_config('Support_Config/config_test.yaml')
# data_manager=get_dataset(config)
# data_loader = DataLoader(data_manager, num_workers=config['data']['workers'])





# data_loader = DataLoader(data_manager, num_workers=config['data']['workers'])

# for i, batch in enumerate(data_loader):
#     # Currently 
#     # batch={dict:5}
#     #   'frame_id'=Tensor(1,)
#     #   'c2w'=Tensor(4,4)
#     #   'rgb'=Tensor(680,1200,3)
#     #   'depth'=Tensor(680,1200)
#     #   'direction'=Tensor(680,1200,3)

#     batch.pop('frame_id')
#     batch['c2w'] = batch['c2w'].squeeze()
#     batch['rgb'] = torch.flatten(batch['rgb'], 0, 2)
#     batch['depth'] = torch.flatten(batch['depth'])
#     batch['direction'] = torch.flatten(batch['direction'], 0, 2)
#     for item in batch:
#         batch[item] = batch[item].squeeze().to('cuda')  # squeeze一下
    
    
#     batch['points']=torch.zeros_like(batch['rgb'])
    
#     memory = torch.zeros((this_offset, F)).data.fill_(0.005)

#     get_xyz(batch['c2w'],batch['direction'],batch['depth'],batch['points']) # CUDA
#     output = torch.zeros((N,L, F), device='cuda', dtype=memory.dtype)
    
    
    
#     input=batch['points'].to('cuda')
#     memory=memory.to('cuda')
#     offsets=offsets.to('cuda')
#     resolution_list=resolution_list.to('cuda')
#     side_length_list=side_length_list.to('cuda')
#     output=output.to('cuda')
#     print('--------------------------------')   
#     for j in range(4):


#         with Timing('forward'):
#             _backend.grid_encode_forward(input, memory, offsets, output,
#                                         resolution_list,side_length_list,
#                                         T)
#         pass
    
    
#         print('output',output.max())
#         print('--------------------------------')    
#     break

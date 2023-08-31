import torch
from get_encoding import get_encoding
import _test as _backend
import math
import numpy as np






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
        print(self.name, 'elapsed', self.start.elapsed_time(self.end), 'ms')






N=2



D = 3
bounds = [ [ -3,3 ],[ -4,2.5 ],[ -2,2.5 ] ]
offsets = []  # placeholder for offsets
resolution_list = []  # placeholder for resolution list
side_length_list = []  # side length of each level's voxels

# variables in paper
L = 16  # number of levels
F = 28 # number of Features
T = 524288  # max entries (hash table size)
N_min = 16
N_max = 512
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



input=torch.randn(N,D) # [N,D]

memory=torch.empty(offsets[-1],F)# [N,F]

output1=torch.empty(N,L,F) #[N,L,F]


input=input.to('cuda')
memory=memory.to('cuda')
offsets=offsets.to('cuda')
resolution_list=resolution_list.to('cuda')
side_length_list=side_length_list.to('cuda')
output1=output1.to('cuda')


with Timing('forward'):
    _backend.grid_encode_forward(input, memory, offsets, output1,
                                resolution_list,side_length_list,
                                T)



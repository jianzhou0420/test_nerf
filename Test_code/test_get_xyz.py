import time
from utils import Timing
import torch
from _mynerf_utils import get_xyz,test





N=8160000
c2w=torch.randn(4,4)
c2w[-1,:]=0
c2w[-1,-1]=1
print('c2w=',c2w.data)

depth=torch.randn(N,1)
# print(depth)
direction=torch.randn(N,3)
point1=torch.randn(N,3)


# print('point0=',point1)

depth=depth.to('cuda')
direction=direction.to('cuda')
c2w=c2w.to('cuda')
point1=point1.to('cuda')

# test()
with Timing('test'):
    get_xyz(c2w,direction,depth,point1)


depth=depth.to('cpu')
direction=direction.to('cpu')
c2w=c2w.to('cpu')
point1=point1.to('cpu')


new_direction=direction.squeeze()*depth
test=torch.cat((new_direction,torch.ones(N,1)),1)

point2=torch.empty(N,3)
with Timing('test2'):
    for i in range(N):
        point2[i,:]=(c2w@test[i,:])[0:3]

print('point1=',point1)
print('point2=',point2)
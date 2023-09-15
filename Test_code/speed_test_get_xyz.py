import time
from utils import Timing
import torch


try:
    from _mynerf_utils import get_xyz,test
except:
    ImportError('Cannot import _mynerf_utils, Please Install "/utils_cuda" first')





N=816000
c2w=torch.randn(4,4,dtype=torch.float,device='cuda')
c2w[-1,:]=0
c2w[-1,-1]=1


depth=torch.randn(N,1,dtype=torch.float,device='cuda')
direction=torch.randn(N,3,dtype=torch.float,device='cuda')
bound=torch.tensor([-3,-4.5,-2.5],dtype=torch.float,device='cuda')




point1=torch.zeros(N,3,dtype=torch.float,device='cuda')

# test()
with Timing('MyCuda'):
    get_xyz(c2w,direction,depth,point1,bound)



new_direction=direction.squeeze()*depth
test=torch.cat((new_direction,torch.ones(N,1,device='cuda')),1)


point2=torch.zeros(N,3,device='cuda',dtype=torch.float)


c2w=c2w.repeat(N,1,1)

with Timing('torch.bmm'):
    point2=torch.bmm(c2w,test.unsqueeze(2)).squeeze(2)[:,0:3]-bound


print('isEqual=',torch.allclose(point1,point2,atol=1e-5,rtol=1e-5))
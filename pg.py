
import torch

def trilinear_interpolation(x, y, z, data):
    x0, y0, z0 = int(x), int(y), int(z)
    x1, y1, z1 = x0 + 1, y0 + 1, z0 + 1
    
    xd = x - x0
    yd = y - y0
    zd = z - z0
    
    c000 = data[x0, y0, z0] * (1 - xd) * (1 - yd) * (1 - zd)
    c001 = data[x0, y0, z1] * (1 - xd) * (1 - yd) * zd
    c010 = data[x0, y1, z0] * (1 - xd) * yd * (1 - zd)
    c011 = data[x0, y1, z1] * (1 - xd) * yd * zd
    
    c100 = data[x1, y0, z0] * xd * (1 - yd) * (1 - zd)
    c101 = data[x1, y0, z1] * xd * (1 - yd) * zd
    c110 = data[x1, y1, z0] * xd * yd * (1 - zd)
    c111 = data[x1, y1, z1] * xd * yd * zd
    
    interpolated_value = (
        c000 + c001 + c010 + c011 + c100 + c101 + c110 + c111
    )
    return interpolated_value


data=torch.zeros((2,2,2,2),dtype=torch.float32)
data[0,0,0,:]=torch.tensor([0,0])
data[0,0,1,:]=torch.tensor([0,0])
data[0,1,0,:]=torch.tensor([0,0])
data[0,1,1,:]=torch.tensor([0,0])
data[1,0,0,:]=torch.tensor([1,1])
data[1,0,1,:]=torch.tensor([1,1])
data[1,1,0,:]=torch.tensor([1,1])
data[1,1,1,:]=torch.tensor([1,1])

x = 0.30769231
y = 0.30769231
z = 0.30769231

interpolated_value = trilinear_interpolation(x, y, z, data)
print(interpolated_value)
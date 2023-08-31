import torch
import torch.nn as nn

class NerfNetwork(nn.Module):
    def __init__(self,encoder_grid,encoder_dir):
        super(NerfNetwork, self).__init__()
        self.L=16,
        self.F=2,
        
        
        # encoder
        self.encoder_grid=encoder_grid
        self.encoder_dir=encoder_dir
        
        # sigma network
        self.input_size=self.L*self.F
        self.num_layers_sigma=1 # TODO: paramize this
        self.hidden_size_sigma=64
        self.output_size_sigma=16
        
        sigma_net=[]
        for layer in range(self.num_layers_sigma):
            if layer==0:
                sigma_net.append(nn.Linear(self.input_size,self.hidden_size_sigma))
                sigma_net.append(nn.ReLU())
            elif layer==self.num_layers_sigma:
                sigma_net.append(nn.Linear(self.hidden_size_sigma,self.output_size_sigma))
            else:
                sigma_net.append(nn.Linear(self.hidden_size_sigma,self.hidden_size_sigma))
                sigma_net.append(nn.ReLU())
                
        self.sigma_net=nn.Sequential(sigma_net)


        # color network
        # TODO: figure out what is the input size actually mean, currently use the same as 
        # TODO: Spherical Harmonics
        
        self.input_size= 16+todo
        self.num_layers_color=2 # TODO: paramize this
        self.hidden_size_color=64
        self.output_size_color=16
        
        color_net=[]
        for layer in range(self.num_layers_color):
            if layer==0:
                color_net.append(nn.Linear(self.input_size,self.hidden_size_color))
                color_net.append(nn.ELU())
            elif layer==self.num_layers_color:
                color_net.append(nn.Linear(self.hidden_size_color,self.output_size_color))
            else:
                color_net.append(nn.Linear(self.hidden_size_color,self.hidden_size_color))
                color_net.append(nn.ELU())
                
        self.color_net=nn.Sequential(color_net)
        
        
        
    def forward(self, x,d):
        # x:[num_points,L*F]
        # d:[num_points,3] the view direction projected onto the first 16 coefficients of the spherical harmonics basis (i.e. up to degree 4). This is a natural frequency encoding over unit vectors.
        
        # sigma
        x=self.encoder(x)

        h=x # I dont know why to copy this, but it is in the original code
        for layer in range(self.num_layers_sigma):
            x=self.sigma_net[layer](x)
        
        sigma=torch.exp(x) # TODO: what is trunc_exp in torch-ngp?
        SH_coeff=x[...,1:]
        
        # color
        d=self.encoder_dir(d) # TODO: implement this, what is it?
        h=torch.cat((SH_coeff,d),dim=-1)
        for layer in range(self.num_layers_color):
            h=self.color_net[layer](h)
        
        # sigmoid activation for rgb
        color = torch.sigmoid(h)
        
        return sigma, color
        
    def density(self,x):
        # x=self.encoder(x)

        h=x # I dont know why to copy this, but it is in the original code
        for layer in range(self.num_layers_sigma):
            x=self.sigma_net[layer](x)
        
        sigma=torch.exp(x) # TODO: what is trunc_exp in torch-ngp?
        SH_coeff=x[...,1:]
        
        return{
            'sigma':sigma,
            'SH_coeff':SH_coeff
        }
    
    def color(self, d, mask=None, SH_coeff=None, **kwargs): #TODO: this is just a copy, implement it by yourself 
        # x: [N, 3] in [-bound, bound]
        # mask: [N,], bool, indicates where we actually needs to compute rgb.

        # if mask is not None:
        #     rgbs = torch.zeros(mask.shape[0], 3, dtype=x.dtype, device=x.device) # [N, 3]
        #     # in case of empty mask
        #     if not mask.any():
        #         return rgbs
        #     x = x[mask]
        #     d = d[mask]
        #     SH_coeff = SH_coeff[mask]

        
        h = torch.cat([d, SH_coeff], dim=-1)
        for l in range(self.num_layers_color):
            h = self.color_net[l](h)
            if l != self.num_layers_color - 1:
                h = nn.functional.relu(h, inplace=True)
        
        # sigmoid activation for rgb
        h = torch.sigmoid(h)

        if mask is not None:
            rgbs[mask] = h.to(rgbs.dtype) # fp16 --> fp32
        else:
            rgbs = h

        return rgbs   

# class baseNeRF(nn.Module):
#     def __init__(self):
#         super(baseNeRF, self).__init__()



class MyNetwork(nn.module):# TODO: give it a suitable name
    def __init__(self,encoder_grid,encoder_dir):
        super(NerfNetwork, self).__init__()
        self.L=16
        self.F=1+9+9+9 # density + SH_coeff_R + SH_coeff_G + SH_coeff_B, sum of the degree of 0,1,2
        
        
        # encoder
        self.encoder_grid=encoder_grid
        
        # sigma network
        self.input_size=self.L*self.F
        self.num_layers_sigma=1 # TODO: paramize this
        self.hidden_size_sigma=64
        self.output_size_sigma=16
        
        sigma_net=[]
        for layer in range(self.num_layers_sigma):
            if layer==0:
                sigma_net.append(nn.Linear(self.input_size,self.hidden_size_sigma))
                sigma_net.append(nn.ReLU())
            elif layer==self.num_layers_sigma:
                sigma_net.append(nn.Linear(self.hidden_size_sigma,self.output_size_sigma))
            else:
                sigma_net.append(nn.Linear(self.hidden_size_sigma,self.hidden_size_sigma))
                sigma_net.append(nn.ReLU())
                
        self.sigma_net=nn.Sequential(sigma_net)



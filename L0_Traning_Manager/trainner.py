import torch
from torch.utils.data import DataLoader
import argparse
from tqdm import tqdm

import torch.optim as optim 

from _mynerf_utils import get_xyz,test
from L1_Data_Manager.dataset import get_dataset
from L2_Encoding_Manager.get_encoding import get_encoding


class Tranner:
    def __init__(self,config):
        self.data_manager = get_dataset(config)
        self.grid_encoding=get_encoding(config['grid']['enc'],config)
        self.SH_encoding=get_encoding('SHEncoder',config)
        self.grid_encoding.to('cuda')
        self.SH_encoding.to('cuda')
        
        self.config=config
        self.device=torch.device('cuda')
        # TODO: set the optimizer
        self.D=config['grid']['input_dimension']
        self.F=9+9+9+1 # SH_R + SH_G + SH_B + density 
        

        
        self.optimizer=optim.Adam(self.grid_encoding.parameters(),lr=0.002)
       
    def train(self):
        data_loader = DataLoader(self.data_manager, num_workers=self.config['data']['workers'])
        print('Data loader is ready.')
        

        
        
        for i, batch in enumerate(data_loader):
            # Currently 
            # batch={dict:5}
            #   'frame_id'=Tensor(1,)
            #   'c2w'=Tensor(4,4)
            #   'rgb'=Tensor(680,1200,3)
            #   'depth'=Tensor(680,1200)
            #   'direction'=Tensor(680,1200,3)
        
            batch.pop('frame_id')
            batch['c2w'] = batch['c2w'].squeeze()
            batch['rgb'] = torch.flatten(batch['rgb'], 0, 2)
            batch['depth'] = torch.flatten(batch['depth'])
            batch['direction'] = torch.flatten(batch['direction'], 0, 2)
            for item in batch:
                batch[item] = batch[item].squeeze()  # squeeze一下
            
            
            batch['points']=torch.empty_like(batch['rgb'])
            get_xyz(batch['c2w'],batch['direction'],batch['depth'],batch['points']) # CUDA
                        
            self.train_one_batch(i,batch)
            
            
            
    def train_one_batch(self,i,batch):
        
        # Step 1: prepare the data
        
        
        
        # batch={dict:5}
        #   'c2w'=Tensor(4,4)
        #   'rgb'=Tensor(680*1200,3)
        #   'depth'=Tensor(680*1200)
        #   'direction'=Tensor(680*1200,3)
        #   'points'=Tensor(680*1200,3)
        
        # Step 2: Forawrd
        
        
        # 2.1: fetch the grid features
        grid_features=self.grid_encoding.forward(batch['points'])
        
        # 2.2 fetch the SH coefficients
        SH_coefficients=self.SH_encoding.forward(batch['direction'].to('cuda'))
        
        # 2.3: render
        # grid_features have shape (680*1200,16*28)
        grid_features=grid_features.view(-1,16,28) # [num_point,Level,Features]
        # for now leave the efficiency problem alone
        R_f=grid_features[:,:,0:9]   # [816000,16,9]
        G_f=grid_features[:,:,9:18]
        B_f=grid_features[:,:,18:27]
        density=grid_features[:,:,27]
        
        
        predicted_rgb=self.render(R_f,G_f,B_f,density,SH_coefficients).view(-1,3)
        
        # Step 3: Backward
        

        self.optimizer.zero_grad()
        loss=torch.nn.functional.mse_loss(predicted_rgb,batch['rgb'].to('cuda'))
        loss.backward()
        self.optimizer.step()
        #
        
        
        
    def render(self,R_f,G_f,B_f,density,SH):
        '''
        R_f: [816000,16,9]
        SH:[816000,9]
        density: [816000,16]
        '''
        # R_f,G_f,B_f,density have shape (680*1200,16,9)
         
        SH=SH.unsqueeze(1).transpose(1,2)
        
        
        density=density.unsqueeze(1)
            
        new_R=torch.bmm(density,torch.bmm(R_f,SH)) # bmm: batch matrix multiply
        new_G=torch.bmm(density,torch.bmm(G_f,SH))
        new_B=torch.bmm(density,torch.bmm(B_f,SH))
        
        
        
        return torch.cat((new_R,new_G,new_B),dim=1).squeeze()
                     
          
    



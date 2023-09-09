import torch
from torch.utils.data import DataLoader
import argparse
from tqdm import tqdm

import torch.optim as optim 

from _mynerf_utils import get_xyz,test
from L1_Data_Manager.dataset import get_dataset
from L2_Encoding_Manager.get_encoding import get_encoding
from torch.optim import lr_scheduler
import matplotlib.pyplot as plt

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
        print('Timming:',self.name, 'elapsed', self.start.elapsed_time(self.end), 'ms')


class Tranner:
    def __init__(self,config):
        self.data_manager = get_dataset(config)
        self.grid_encoding=get_encoding(config['grid']['enc'])
        self.SH_encoding=get_encoding('SHEncoder')
        self.grid_encoding.to('cuda')
        self.SH_encoding.to('cuda')
        
        self.config=config
        self.device=torch.device('cuda')
        # TODO: set the optimizer
        self.D=config['grid']['input_dimension']
        self.F=9+9+9+1 # SH_R + SH_G + SH_B + density 
        
        self.optimizer=optim.Adam(self.grid_encoding.parameters(),lr=0.05)
        self.scheduler=lr_scheduler.StepLR(self.optimizer,step_size=100,gamma=0.3)
        self.lr_change=False
       
      
    def render(self,R_f,G_f,B_f,density,SH):
        '''
        R_f: [816000,16,9]
        SH:[816000,9]
        density: [816000,16]
         '''
        # R_f,G_f,B_f,density have shape (680*1200,16,9)
        
        # first we need to calcualte rgb by R_f and SH
        test1=(R_f*SH.unsqueeze(1)).sum(dim=2) # [816000,16]
        test2=(G_f*SH.unsqueeze(1)).sum(dim=2) # [816000,16]
        test3=(B_f*SH.unsqueeze(1)).sum(dim=2) # [816000,16]
        test11=(test1*density).sum(dim=1) # [816000]
        test22=(test2*density).sum(dim=1) # [816000]
        test33=(test3*density).sum(dim=1) # [816000]
        rgb=torch.stack([test11,test22,test33],dim=1) # [816000,3]
        
        
        # one layer
        # R_f=R_f[:,-1,:].squeeze()
        # G_f=G_f[:,-1,:].squeeze()
        # B_f=B_f[:,-1,:].squeeze()
        
        # R=(R_f*SH).sum(dim=1) # [816000]
        # G=(G_f*SH).sum(dim=1) # [816000]
        # B=(B_f*SH).sum(dim=1) # [816000]
        
        # rgb=torch.stack([R,G,B],dim=1) # [816000,3]
        return rgb
                     
    def render_from_pt(self):
        '''
        This function is used to render from the checkpoint. You have to set the c2w matrix and ray direction.
        you have to write a ray marching function.
        '''
        self.grid_encoding.memory=torch.load('./testall2.pt')
        data_loader = DataLoader(self.data_manager, num_workers=self.config['data']['workers'])
        for i, batch in enumerate(data_loader):
            batch.pop('frame_id')
            batch['c2w'] = batch['c2w'].squeeze()
            batch['rgb'] = torch.flatten(batch['rgb'], 0, 2)
            batch['depth'] = torch.flatten(batch['depth'])
            batch['direction'] = torch.flatten(batch['direction'], 0, 2)
            for item in batch:
                batch[item] = batch[item].squeeze().to('cuda')  # squeeze一下
            
            
            batch['points']=torch.empty_like(batch['rgb'])
            with Timing('Last render'):
                get_xyz(batch['c2w'],batch['direction'],batch['depth'],batch['points'],torch.tensor([-3,-4.5,-2.5],dtype=torch.float32,device='cuda')) # CUDA
            
            
            # with Timing('grid'):
                grid_features=self.grid_encoding.forward(batch['points'])
            
            # with Timing('SH'):
                SH_coefficients=self.SH_encoding.forward(batch['direction'])

            # with Timing('render'):
                # 2.3: render
                # grid_features have shape (680*1200,16*28)
                grid_features=grid_features.view(-1,self.grid_encoding.L,self.grid_encoding.F) # [num_point,Level,Features]
                # for now leave the efficiency problem alone
                R_f=grid_features[:,:,0:9]   # [816000,16,9]
                G_f=grid_features[:,:,9:18]
                B_f=grid_features[:,:,18:27]
                density=grid_features[:,:,27]
                predicted_rgb=self.render(R_f,G_f,B_f,density,SH_coefficients).view(-1,3) # TODO: integrate the grid and SH
                predicted_rgb=torch.clamp(predicted_rgb,0,1)
            rgb=predicted_rgb.to('cpu').detach().numpy()
            rgb=rgb.reshape(680,1200,3)
            plt.imshow(rgb)
            plt.show()
            plt.savefig('test.png')
        
        

    def train(self):
        data_loader = DataLoader(self.data_manager, num_workers=self.config['data']['workers'])

        for epoch in range(20):
            
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
                    batch[item] = batch[item].squeeze().to('cuda')  # squeeze一下
                
                
                batch['points']=torch.empty_like(batch['rgb'])
                # with Timing('get_xyz'):
                get_xyz(batch['c2w'],batch['direction'],batch['depth'],batch['points'],torch.tensor([-3,-4.5,-2.5],dtype=torch.float32,device='cuda')) # CUDA
                
                batch['points']=batch['points']
                self.loss1=[]
                
            
    
                self.train_one_batch(batch['c2w'],batch['rgb'],batch['depth'],batch['direction'],batch['points'])
                print('This is the number',i,'/2000 iteration,',epoch,'/20')
        
        torch.save(self.grid_encoding.memory,'./testall.pt')
        
        data_loader = DataLoader(self.data_manager, num_workers=self.config['data']['workers'])
        # test
        
        
        for i, batch in enumerate(data_loader):
            batch.pop('frame_id')
            batch['c2w'] = batch['c2w'].squeeze()
            batch['rgb'] = torch.flatten(batch['rgb'], 0, 2)
            batch['depth'] = torch.flatten(batch['depth'])
            batch['direction'] = torch.flatten(batch['direction'], 0, 2)
            for item in batch:
                batch[item] = batch[item].squeeze().to('cuda')  # squeeze一下
            
            
            batch['points']=torch.empty_like(batch['rgb'])
            with Timing('Last render'):
                get_xyz(batch['c2w'],batch['direction'],batch['depth'],batch['points'],torch.tensor([-3,-4.5,-2.5],dtype=torch.float32,device='cuda')) # CUDA
            
           
            # with Timing('grid'):
                grid_features=self.grid_encoding.forward(batch['points'])
            
            # with Timing('SH'):
                SH_coefficients=self.SH_encoding.forward(batch['direction'])

            # with Timing('render'):
                # 2.3: render
                # grid_features have shape (680*1200,16*28)
                grid_features=grid_features.view(-1,self.grid_encoding.L,self.grid_encoding.F) # [num_point,Level,Features]
                # for now leave the efficiency problem alone
                R_f=grid_features[:,:,0:9]   # [816000,16,9]
                G_f=grid_features[:,:,9:18]
                B_f=grid_features[:,:,18:27]
                density=grid_features[:,:,27]
                predicted_rgb=self.render(R_f,G_f,B_f,density,SH_coefficients).view(-1,3) # TODO: integrate the grid and SH
                predicted_rgb=torch.clamp(predicted_rgb,0,1)
            break
        rgb=predicted_rgb.to('cpu').detach().numpy()
        rgb=rgb.reshape(680,1200,3)
        plt.imshow(rgb)
        plt.show()
        plt.savefig('test.png')
            



    def train_one_batch(self,c2w,rgb,depth,direction,points):
        
        with Timing('One Batch'):
            grid_features=self.grid_encoding.forward(points)
           
        # with Timing('SH'):
            SH_coefficients=self.SH_encoding.forward(direction)

        # with Timing('render'):
            # 2.3: render
            # grid_features have shape (680*1200,16*28)
            grid_features=grid_features.view(-1,self.grid_encoding.L,self.grid_encoding.F) # [num_point,Level,Features]
            # for now leave the efficiency problem alone
            R_f=grid_features[:,:,0:9]   # [816000,16,9]
            G_f=grid_features[:,:,9:18]
            B_f=grid_features[:,:,18:27]
            density=grid_features[:,:,27]
            
            
            predicted_rgb=self.render(R_f,G_f,B_f,density,SH_coefficients).view(-1,3) # TODO: integrate the grid and SH
            predicted_rgb=torch.clamp(predicted_rgb,0,1)
            # print('predicted_rgb',predicted_rgb.max())
            # print('rgb',rgb.max())
            
        # with Timing('loss'):
            
            loss=torch.nn.functional.mse_loss(predicted_rgb,rgb)
            # print('loss:',loss)
        
            loss.backward()
            
            self.optimizer.step()
            if self.lr_change:
                self.scheduler.step()
                
            self.optimizer.zero_grad()
        self.loss1.append(loss.to('cpu').detach().numpy())
        
        
        # print('test finished.')
        # print('------------------------------------')

        
        
        
        
        
        
        
        
        
        
        
        
        
        
            # def train(self):
    #     data_loader = DataLoader(self.data_manager, num_workers=self.config['data']['workers'])
        
    #     for i, batch in tqdm(enumerate(data_loader)):
    #         # Currently 
    #         # batch={dict:5}
    #         #   'frame_id'=Tensor(1,)
    #         #   'c2w'=Tensor(4,4)
    #         #   'rgb'=Tensor(680,1200,3)
    #         #   'depth'=Tensor(680,1200)
    #         #   'direction'=Tensor(680,1200,3)
        
    #         batch.pop('frame_id')
    #         batch['c2w'] = batch['c2w'].squeeze()
    #         batch['rgb'] = torch.flatten(batch['rgb'], 0, 2)
    #         batch['depth'] = torch.flatten(batch['depth'])
    #         batch['direction'] = torch.flatten(batch['direction'], 0, 2)
    #         for item in batch:
    #             batch[item] = batch[item].squeeze().to('cuda')  # squeeze一下
            
            
    #         batch['points']=torch.empty_like(batch['rgb'])
    #         get_xyz(batch['c2w'],batch['direction'],batch['depth'],batch['points']) # CUDA

    #         for j in range(30):
    #             self.train_one_batch(i,batch)
    #         if i==5:break
            
    #     torch.save(self.grid_encoding.memory,'./memory.pt')
    #     # print('Memory saved.')
    #     # print('Training finished.')
            
    # def train_one_batch(self,i,batch):
        
    #     # Step 1: prepare the data
    #     # batch={dict:5}
    #     #   'c2w'=Tensor(4,4)
    #     #   'rgb'=Tensor(680*1200,3)
    #     #   'depth'=Tensor(680*1200)
    #     #   'direction'=Tensor(680*1200,3)
    #     #   'points'=Tensor(680*1200,3)
        
    #     # Step 2: Forawrd
    #     # 2.1: fetch the grid features
    #     grid_features=self.grid_encoding.forward(batch['points'])
        
    #     # 2.2 fetch the SH coefficients
    #     SH_coefficients=self.SH_encoding.forward(batch['direction'])
        
    #     # 2.3: render
    #     # grid_features have shape (680*1200,16*28)
    #     grid_features=grid_features.view(-1,16,28) # [num_point,Level,Features]
    #     # for now leave the efficiency problem alone
    #     R_f=grid_features[:,:,0:9]   # [816000,16,9]
    #     G_f=grid_features[:,:,9:18]
    #     B_f=grid_features[:,:,18:27]
    #     density=grid_features[:,:,27]
        
        
    #     predicted_rgb=self.render(R_f,G_f,B_f,density,SH_coefficients).view(-1,3) # TODO: integrate the grid and SH
        
    #     # Step 3: Backward
        
    #     self.scheduler.step()
    #     self.optimizer.zero_grad()
    #     loss=torch.nn.functional.mse_loss(predicted_rgb,batch['rgb'].to('cuda'))
    #     loss.backward()
    #     self.optimizer.step()
    #     # print('loss:',loss)
        
        
  
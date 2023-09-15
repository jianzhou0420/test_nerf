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
import numpy as np
import cv2
import os

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
        
        self.optimizer=optim.Adam(self.grid_encoding.parameters(),lr=0.1)
        self.scheduler=lr_scheduler.StepLR(self.optimizer,step_size=2,gamma=0.2)
        self.lr_change=False
       
      

    def train(self):
        data_loader = DataLoader(self.data_manager, num_workers=self.config['data']['workers'])

        # for epoch in range(20):
            
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
            
        
            # for j in range(20):
            self.train_one_batch(batch['rgb'],batch['direction'],batch['points'])
            print('This is the number','/20 iteration,',i,'/2000 batch')
    
        torch.save(self.grid_encoding.memory,'./testall10.pt')
        
        
            
    def train_one_batch(self,rgb,direction,points):
        
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

        # with Timing('loss'):
            loss=torch.nn.functional.mse_loss(predicted_rgb,rgb)
            loss.backward()
            self.optimizer.step()   
            self.optimizer.zero_grad()
            
            if self.lr_change:
                self.scheduler.step()
            print('lr',self.optimizer.param_groups[0]['lr'])
        self.loss1.append(loss.to('cpu').detach().numpy())
        
    def render(self,R_f,G_f,B_f,density,SH):
        '''
        R_f: [816000,16,9]
        SH:[816000,9]
        density: [816000,16]
         '''
        # R_f,G_f,B_f,density have shape (680*1200,16,9)
        
        # first we need to calcualte rgb by R_f and SH
        R_layer=(R_f*SH.unsqueeze(1)).sum(dim=2) # [816000,16]
        G_layer=(G_f*SH.unsqueeze(1)).sum(dim=2) # [816000,16]
        B_layer=(B_f*SH.unsqueeze(1)).sum(dim=2) # [816000,16]
        R=(R_layer*density).sum(dim=1) # [816000]
        G=(G_layer*density).sum(dim=1) # [816000]
        B=(B_layer*density).sum(dim=1) # [816000]
        
        rgb=torch.stack([R,G,B],dim=1) # [816000,3]
            
        return rgb
                     
    def render_from_pt(self,pt_path,save_path):
        '''
        This function is used to render from the checkpoint. You have to set the c2w matrix and ray direction.
        you have to write a ray marching function.
        '''
        
        assert type(pt_path)==str, 'pt should be a string'
        
        try :
            self.grid_encoding.memory=torch.load(pt_path)
        except:
            print('Cannot load the checkpoint')
            return
        
        os.makedirs(save_path, exist_ok=True)
        
        data_loader = DataLoader(self.data_manager, num_workers=self.config['data']['workers'])
        for i, batch in enumerate(data_loader):
            batch.pop('frame_id')
            batch['c2w'] = batch['c2w'].squeeze()
            batch['rgb'] = torch.flatten(batch['rgb'], 0, 2)
            batch['depth'] = torch.flatten(batch['depth'])
            batch['direction'] = torch.flatten(batch['direction'], 0, 2)
            
            for item in batch:# squeeze and send to cuda
                batch[item] = batch[item].squeeze().to('cuda') 
            
            
            batch['points']=torch.empty_like(batch['rgb'])
            with Timing('Last render'):
                get_xyz(batch['c2w'],batch['direction'],batch['depth'],batch['points'],torch.tensor([-3,-4.5,-2.5],dtype=torch.float32,device='cuda')) # CUDA
            
            

                grid_features=self.grid_encoding.forward(batch['points'])
        
                SH_coefficients=self.SH_encoding.forward(batch['direction'])

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
                rgb=(rgb.reshape(680,1200,3)*255).astype(np.uint8)

                file_name = os.path.join(save_path, f"output_{i}.png")
                
            
            # with Timing('plt'):
                cv2.imwrite(file_name, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))


    def PSNR(self,rendered_image_path,GT_image_path):

        # Load the original and processed images
        original_image = cv2.imread(GT_image_path)  # Load your original image
        processed_image = cv2.imread(rendered_image_path)  # Load your processed image

        # Ensure that both images have the same dimensions
        if original_image.shape != processed_image.shape:
            raise ValueError("Both images should have the same dimensions.")

        # Calculate the Mean Squared Error (MSE) between the original and processed images
        mse = np.mean((original_image - processed_image) ** 2)

        # Calculate the maximum possible pixel value (assuming an 8-bit image)
        max_pixel_value = 255.0

        # Calculate the PSNR
        psnr = 10 * np.log10((max_pixel_value ** 2) / mse)

        print(f"PSNR: {psnr} dB")
        pass
    
    def test_one_picture(self):
        data_loader = DataLoader(self.data_manager, num_workers=self.config['data']['workers'])
        # for epoch in range(10):
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
            
        
            for j in range(300):
                self.train_one_batch(batch['rgb'],batch['direction'],batch['points'])
                print('This is the number',j,'/30 iteration',i,'/30 batch','/30 epoch')
            break
            
        # plt.plot(self.loss1)
        # plt.xlabel('iteration')
        # plt.ylabel('loss')
        
        plt.show()
        for i, batch in enumerate(data_loader):
            batch.pop('frame_id')
            batch['c2w'] = batch['c2w'].squeeze()
            batch['rgb'] = torch.flatten(batch['rgb'], 0, 2)
            batch['depth'] = torch.flatten(batch['depth'])
            batch['direction'] = torch.flatten(batch['direction'], 0, 2)
            
            for item in batch:# squeeze and send to cuda
                batch[item] = batch[item].squeeze().to('cuda') 
            
            
            batch['points']=torch.empty_like(batch['rgb'])
            with Timing('Last render'):
                get_xyz(batch['c2w'],batch['direction'],batch['depth'],batch['points'],torch.tensor([-3,-4.5,-2.5],dtype=torch.float32,device='cuda')) # CUDA
            
            

                grid_features=self.grid_encoding.forward(batch['points'])
        
                SH_coefficients=self.SH_encoding.forward(batch['direction'])

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
                rgb=(rgb.reshape(680,1200,3)*255).astype(np.uint8)
                # plt.imshow(rgb)
                # plt.show()
                save_path='./'
                file_name = os.path.join(save_path, f"output_{i}.png")
                
            
            # with Timing('plt'):
                cv2.imwrite(file_name, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
                PSNR=self.PSNR(file_name,"/home/jian/nerf/mynerf/L1_Data_Manager/Replica/office0/results/frame000000.jpg")
                print('PSNR',PSNR)
                break
                # break
        
        
        
        
        
        
        
        
        
        
        
        
        
        
 
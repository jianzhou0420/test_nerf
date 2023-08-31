# outer imports
import torch
from torch.utils.data import DataLoader
import argparse
from tqdm import tqdm
# inner imports
from L1_Data_Manager.dataset import get_dataset

from utils import load_config
from L2_Encoding_Manager.get_encoding import get_map





class Tranner:
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda')
        self.dataset_manager = get_dataset(config)

        self.map_manager = get_map(config['grid']['enc'], config)
        # self.HashManager = HashManager()
        # self.Tracking = Tracking(self.dataset, config)
        # self.Voxels = Voxels()
        # self.Trainer = Trainer()

    def run(self,load_batch=True):
        # self.map_manager.run()
        data_loader = DataLoader(self.dataset_manager, num_workers=self.config['data']['workers'])
        print('Data loader is ready.')
        for i, batch in enumerate(data_loader):
            # 好，到现在为止，数据已经可以读取了，tracking不做的情况下，下面需要的就是最重要的mapping的部分，用explicit的方法来mapping
            # batch={dict:5}
            #   'framee_id'=Tensor(1,)
            #   'c2w'=Tensor(4,4)
            #   'rgb'=Tensor(680,1200,3)
            #   'depth'=Tensor(680,1200)
            #   'direction'=Tensor(680,1200,3)
            # do some preprocess on batch to get a more easy format
            if load_batch:
                batch = torch.load('/home/jian/nerf/mynerf/Test_Data/batch_test.pt')
            else:
            
                batch['frame_id'] = batch['frame_id'].squeeze()
                batch['c2w'] = batch['c2w'].squeeze()
                batch['rgb'] = torch.flatten(batch['rgb'], 0, 2)
                batch['depth'] = torch.flatten(batch['depth'])
                batch['direction'] = torch.flatten(batch['direction'], 0, 2)
                for item in batch:
                    batch[item] = batch[item].squeeze()  # squeeze一下
                # Now batch={dict:5}
                #   'frame_id'=Tensor(1)
                #   'c2w'=Tensor(4,4)
                #   'rgb'=Tensor(680*1200,3)
                #   'depth'=Tensor(680*1200)
                #   'direction'=Tensor(680*1200,3)

                points = torch.empty((batch['depth'].numel(), config['grid']['input_dimension']))

                for i in range(batch['depth'].numel()):
                    print(i)
                    points[i, :] = self.get_ray_coordinate(batch['c2w'], batch['direction'][i, :], batch['depth'][i])

                batch.pop('c2w')
                batch.pop('direction')
                batch.pop('frame_id')
                batch['points'] = points
                torch.save(batch, '/home/jian/nerf/mynerf/test_data/batch_test.pt')
                
            
            #  Now we have prepared the data for mapping
            #  batch={dict:3}
            #  'rgb'=Tensor(680*1200,3)
            #  'depth'=Tensor(680*1200)
            #  'points'=Tensor(680*1200,3)
            for item in batch:
                batch[item] = batch[item].to(self.device)
            this_output=self.map_manager.forward(batch['points'])
            print(this_output.shape)
            break # for test, we only run one batch
            
            

    def get_ray_coordinate(self, c2w, ray_direction, ray_depth):
        '''
        First calculate the relative position the point
        and then convert it to homography representation
        and then appy c2w to it

        note the shape of them
        :param c2w: 4x4
        :param ray_direction: 3xN
        :param ray_depth: 1xN
        :return:
        '''
        point2world = (c2w @ torch.cat((ray_direction * ray_depth, torch.tensor([1]))))[0:3]
        return point2world


if __name__ == '__main__':
    print('System is starting...')

    parser = argparse.ArgumentParser(
        description='ENMapping'
    )
    parser.add_argument('--config', type=str, help='path to config file')
    parser.add_argument('--input_folder', type=str, help='path to input folder')
    parser.add_argument('--output_folder', type=str, help='path to output folder')

    args = parser.parse_args()

    args.config = 'Support_Config/config_test.yaml'
    args.input_folder = 'datasets/replica'
    config = load_config(args.config)

    slam = Tranner(config)
    slam.run()

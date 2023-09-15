# outer imports
import torch
from torch.utils.data import DataLoader
import argparse
from tqdm import tqdm


# inner imports
from L0_Traning_Manager.trainner import Tranner
from utils import load_config





print('System is starting...')

parser = argparse.ArgumentParser(
    description='FullyExplicitNeRF'
)
parser.add_argument('--config', type=str, help='path to config file')
parser.add_argument('--input_folder', type=str, help='path to input folder')
parser.add_argument('--output_folder', type=str, help='path to output folder')

args = parser.parse_args()

args.config = 'Support_Config/config_test.yaml'
args.input_folder = 'datasets/replica'
config = load_config(args.config)

test = Tranner(config)

# test.train()
# test.render_from_pt('./testall10.pt','./Test_output/20iteration_2000batch_0.1/')
test.test_one_picture()
# test.PSNR("/home/jian/nerf/mynerf/Test_output/30batch_30epoch0.1lr.png","/home/jian/nerf/mynerf/L1_Data_Manager/Replica/office0/results/frame000000.jpg")
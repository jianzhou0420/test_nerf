from .network.network import *
from L2_Encoding_Manager.get_encoding import get_encoding

# TODO: complete this function

def get_network(type,config):
    if type == 'nerf':
        return NerfNetwork(config)

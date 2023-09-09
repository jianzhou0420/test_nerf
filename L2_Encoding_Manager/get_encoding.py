from .hash_grid.HashGrid import HashGrid
from .sh_encoder import SHEncoder


def get_encoding(type):
    if type == 'HashGrid':
        return HashGrid()
    
    if type =='SHEncoder':
        return SHEncoder(degree=3)

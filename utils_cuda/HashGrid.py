import torch
import torch.nn as nn
from torch.autograd import Function
from torch.cuda.amp import custom_bwd, custom_fwd
import math
import numpy as np

from .backend import _backend


print('yes')
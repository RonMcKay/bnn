import torch.nn as nn

class BayesianLayer(nn.Module):
    def __init__(self):
        super().__init__()

from .utils import KLLoss, Sequential
from .layers import BConv1d, BConv2d, BLinear
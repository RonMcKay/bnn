import torch.nn as nn


class BayesianLayer(nn.Module):
    def __init__(self):
        super().__init__()


from .utils import KLLoss, Sequential, BayesNetWrapper  # noqa: F401
from .layers import BLinear, BConv1d, BConv2d, BConvTranspose1d, BConvTranspose2d  # noqa: F401

import torch.nn as nn


class BayesianLayer(nn.Module):
    def __init__(self):
        super().__init__()


from .layers import (  # noqa: F401
    BConv1d,
    BConv2d,
    BConvTranspose1d,
    BConvTranspose2d,
    BLinear,
)
from .utils import BayesNetWrapper, KLLoss, Sequential  # noqa: F401

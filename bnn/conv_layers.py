import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _single, _pair, _triple

class _ConvNd(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, prior, var_dist,
                 stride, padding, dilation, transposed, output_padding,
                 groups, bias, padding_mode):
        super().__init__()
        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.transposed = transposed
        self.output_padding = output_padding
        self.groups = groups
        self.padding_mode = padding_mode
        
    def reset_parameters(self):
        raise NotImplementedError('Has to be implemented!')
    
    def extra_repr(self):
        return nn.Module.extra_repr(self)
    
class Conv1d(_ConvNd):
    """Applies a 1D Bayesian convolution over an input signal composed of several input planes.
    """
    def __init__(self, in_channels, out_channels, kernel_size, prior, var_dist,
                 stride=1, padding=0, dilation=1, groups=1,
                 bias=True, padding_mode='zeros'):
        kernel_size = _single(kernel_size)
        stride = _single(stride)
        padding = _single(padding)
        dilation = _single(dilation)
        super().__init__(in_channels, out_channels, kernel_size, prior, var_dist,
                         stride, padding, dilation, False, _single(0), groups,
                         bias, padding_mode)
        
    def forward(self, input):
        raise NotImplementedError('Has to be implemented!')
    
class Conv2d(_ConvNd):
    """Applies a 2d Bayesian convolution over an input signal composed of several input planes.
    """
    def __init__(self, in_channels, out_channels, kernel_size, prior, var_dist,
                 stride=1, padding=0, dilation=1, groups=1,
                 bias=True, padding_mode='zeros'):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super().__init__(in_channels, out_channels, kernel_size, prior, var_dist,
                         stride, padding, dilation, False, _pair(0), groups,
                         bias, padding_mode)
        
    def forward(self, input):
        raise NotImplementedError('Has to be implemented!')
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _single, _pair
import logging

from bnn import BayesianLayer
from bnn.utils import kldivergence
from bnn.distributions.priors import GaussianMixture
from bnn.distributions.posteriors import VariationalDistribution
from bnn.distributions.posteriors import DiagonalNormal


class _BConvNd(BayesianLayer):
    def __init__(self, in_channels, out_channels, kernel_size,
                 weight_prior, weight_posterior, bias_prior, bias_posterior,
                 stride, padding, dilation, transposed, output_padding,
                 groups, bias):
        super().__init__()
        self.log = logging.getLogger(__name__[:__name__.rfind('.')] + '.' + type(self).__name__)
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
        self.bias = bias
        self.weight_prior = weight_prior
        self.weight_posterior = weight_posterior
        self.bias_prior = bias_prior
        self.bias_posterior = bias_posterior

        self._init_default_distributions()

    def extra_repr(self):
        s = ('{in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.output_padding != (0,) * len(self.output_padding):
            s += ', output_padding={output_padding}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.bias is None:
            s += ', bias=False'

        return s.format(**self.__dict__)

    def _init_default_distributions(self):
        # specify default priors and variational posteriors
        dists = {
            'weight_prior': GaussianMixture(sigma1=0.1, sigma2=0.0005, pi=0.75),
            'weight_posterior': DiagonalNormal(
                mean=torch.Tensor(self.out_channels if not self.transposed else self.in_channels,
                                  self.in_channels if not self.transposed else self.out_channels,
                                  *self.kernel_size).normal_(0, 0.1),
                rho=torch.Tensor(self.out_channels if not self.transposed else self.in_channels,
                                 self.in_channels if not self.transposed else self.out_channels,
                                 *self.kernel_size).normal_(-5, 0.1))
        }
        if self.bias:
            dists['bias_prior'] = GaussianMixture(sigma1=0.1, sigma2=0.0005, pi=0.75)
            dists['bias_posterior'] = DiagonalNormal(mean=torch.Tensor(self.out_channels).normal_(0, 0.1),
                                                     rho=torch.Tensor(self.out_channels).normal_(-5, 0.1))

        # specify all distributions that are not given by the user as the default distribution
        for d in dists:
            if getattr(self, d) is None:
                setattr(self, d, dists[d])


class BConv1d(_BConvNd):
    """Applies a 1d Bayesian convolution over an input signal composed of several input planes.
    All Parameters are the same as in the standard pytorch convolution operators, except:
    
    Args:
        weight_prior (nn.Module): Module that has a log_prob function to get logarithmic probabilities. Parameters
                                  should be registered as buffers in order to not optimize them. If ``None``, takes
                                  the gaussian mixture model from 'Weight Uncertainty in Neural Networks' as prior.
                                  Default: ``None``
        weight_posterior (VariationalDistribution): Variational Distribution of the weights to approximate the real
                                                    Posterior Distribution. If ``None``, takes a diagonal gaussian
                                                    distribution as variational distribution. Default: ``None``
        bias_prior (nn.Module): Module that has a log_prob function to get logarithmic probabilities. Parameters
                                should be registered as buffers in order to not optimize them. If ``None``, takes
                                the gaussian mixture model from 'Weight Uncertainty in Neural Networks' as prior.
                                Default: ``None``
        bias_posterior (VariationalDistribution): Variational Distribution of the weights to approximate the real
                                                  Posterior Distribution. If ``None``, takes a diagonal gaussian
                                                  distribution as variational distribution. Default: ``None``
        
    """

    def __init__(self, in_channels, out_channels, kernel_size,
                 weight_prior=None, weight_posterior=None, bias_prior=None, bias_posterior=None,
                 stride=1, padding=0, dilation=1, groups=1,
                 bias=True, padding_mode='zeros'):
        self.log = logging.getLogger(__name__[:__name__.rfind('.')] + '.' + type(self).__name__)
        kernel_size = _single(kernel_size)
        stride = _single(stride)
        padding = _single(padding)
        dilation = _single(dilation)
        super().__init__(in_channels, out_channels, kernel_size,
                         weight_prior, weight_posterior, bias_prior, bias_posterior,
                         stride, padding, dilation, False, _single(0), groups,
                         bias)

    def forward(self, x, **kwargs):
        weights = self.weight_posterior.sample()

        if self.bias:
            bias = self.bias_posterior.sample()
        else:
            bias = None

        out = F.conv1d(x, weight=weights, bias=bias,
                       stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups)

        kl = kldivergence(self.weight_prior, self.weight_posterior, weights)
        if self.bias:
            kl = kl + kldivergence(self.bias_prior, self.bias_posterior, bias)

        return out, kl


class BConv2d(_BConvNd):
    """Applies a 2d Bayesian convolution over an input signal composed of several input planes.
    All Parameters are the same as in the standard pytorch convolution operators, except:
    
    Args:
        weight_prior (nn.Module): Module that has a log_prob function to get logarithmic probabilities. Parameters
                                  should be registered as buffers in order to not optimize them. If ``None``, takes
                                  the gaussian mixture model from 'Weight Uncertainty in Neural Networks' as prior.
                                  Default: ``None``
        weight_posterior (VariationalDistribution): Variational Distribution of the weights to approximate the real
                                                    Posterior Distribution. If ``None``, takes a diagonal gaussian
                                                    distribution as variational distribution. Default: ``None``
        bias_prior (nn.Module): Module that has a log_prob function to get logarithmic probabilities. Parameters
                                should be registered as buffers in order to not optimize them. If ``None``, takes
                                the gaussian mixture model from 'Weight Uncertainty in Neural Networks' as prior.
                                Default: ``None``
        bias_posterior (VariationalDistribution): Variational Distribution of the weights to approximate the real
                                                  Posterior Distribution. If ``None``, takes a diagonal gaussian
                                                  distribution as variational distribution. Default: ``None``
    """

    def __init__(self, in_channels, out_channels, kernel_size,
                 weight_prior=None, weight_posterior=None, bias_prior=None, bias_posterior=None,
                 stride=1, padding=0, dilation=1, groups=1,
                 bias=True):
        self.log = logging.getLogger(__name__[:__name__.rfind('.')] + '.' + type(self).__name__)
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super().__init__(in_channels,
                         out_channels,
                         kernel_size,
                         weight_prior, weight_posterior,
                         bias_prior, bias_posterior,
                         stride,
                         padding,
                         dilation,
                         False,
                         _pair(0),
                         groups,
                         bias)

    def forward(self, x, **kwargs):
        weights = self.weight_posterior.sample()

        if self.bias:
            bias = self.bias_posterior.sample()
        else:
            bias = None

        out = F.conv2d(x, weight=weights, bias=bias,
                       stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups)

        kl = kldivergence(self.weight_prior, self.weight_posterior, weights)

        if self.bias:
            kl = kl + kldivergence(self.bias_prior, self.bias_posterior, bias)

        return out, kl


class _BConvTransposeNd(_BConvNd):
    def __init__(self, in_channels, out_channels, kernel_size,
                 weight_prior, weight_posterior, bias_prior, bias_posterior,
                 stride, padding, dilation, transposed, output_padding, groups,
                 bias):
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            weight_prior,
            weight_posterior,
            bias_prior,
            bias_posterior,
            stride,
            padding,
            dilation,
            transposed,
            output_padding,
            groups,
            bias
        )


class BConvTranspose1d(_BConvTransposeNd):
    def __init__(self, in_channels, out_channels, kernel_size,
                 weight_prior=None, weight_posterior=None, bias_prior=None, bias_posterior=None,
                 stride=1, padding=0, dilation=1, groups=1,
                 bias=True):
        kernel_size = _single(kernel_size)
        stride = _single(stride)
        padding = _single(padding)
        dilation = _single(dilation)
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            weight_prior,
            weight_posterior,
            bias_prior,
            bias_posterior,
            stride,
            padding,
            dilation,
            True,
            _single(0),
            groups,
            bias
        )

    def forward(self, x, **kwargs):
        weights = self.weight_posterior.sample()

        if self.bias:
            bias = self.bias_posterior.sample()
        else:
            bias = None

        out = F.conv_transpose1d(x, weight=weights, bias=bias,
                                 stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups)

        kl = kldivergence(self.weight_prior, self.weight_posterior, weights)

        if self.bias:
            kl = kl + kldivergence(self.bias_prior, self.bias_posterior, bias)

        return out, kl


class BConvTranspose2d(_BConvTransposeNd):
    def __init__(self, in_channels, out_channels, kernel_size,
                 weight_prior=None, weight_posterior=None, bias_prior=None, bias_posterior=None,
                 stride=1, padding=0, dilation=1, groups=1,
                 bias=True):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            weight_prior,
            weight_posterior,
            bias_prior,
            bias_posterior,
            stride,
            padding,
            dilation,
            True,
            _pair(0),
            groups,
            bias
        )

    def forward(self, x, **kwargs):
        weights = self.weight_posterior.sample()

        if self.bias:
            bias = self.bias_posterior.sample()
        else:
            bias = None

        out = F.conv_transpose2d(x, weight=weights, bias=bias,
                                 stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups)

        kl = kldivergence(self.weight_prior, self.weight_posterior, weights)

        if self.bias:
            kl = kl + kldivergence(self.bias_prior, self.bias_posterior, bias)

        return out, kl

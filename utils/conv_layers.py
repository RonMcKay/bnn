import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _single, _pair
import logging

from bnn.utils.general import BayesianLayer
from bnn.utils.priors import DiagonalNormal as PriorNormal
from bnn.utils.priors import GaussianMixture
from bnn.utils.posteriors import VariationalDistribution
from bnn.utils.posteriors import DiagonalNormal


class _BConvNd(BayesianLayer):
    def __init__(self, in_channels, out_channels, kernel_size,
                 weight_prior, weight_posterior, bias_prior, bias_posterior,
                 stride, padding, dilation, transposed, output_padding,
                 groups, bias):
        super().__init__()
        self.log = logging.getLogger(__name__ + 'BConvNd')
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
        self.kl_weights_closed = False
        self.kl_bias_closed = False

        self._init_default_distributions()

    def reset_parameters(self):
        raise NotImplementedError('Has to be implemented!')

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
        x = self.kernel_size[0]
        for i in range(1, len(self.kernel_size)):
            x *= self.kernel_size[i]
        self.log.debug('Number of filter weights: {}'.format(x))
        bound = math.sqrt(2.0 / (x * self.in_channels))
        dists = {'weight_prior': GaussianMixture(sigma1=0.1, sigma2=0.0005, pi=0.75),
                 'weight_posterior': DiagonalNormal(mean=torch.Tensor(self.out_channels,
                                                                      self.in_channels,
                                                                      *self.kernel_size).uniform_(-bound, bound),
                                                    rho=torch.Tensor(self.out_channels,
                                                                     self.in_channels,
                                                                     *self.kernel_size).normal_(-9, 0.001))}
        if self.bias:
            dists['bias_prior'] = GaussianMixture(sigma1=0.1, sigma2=0.0005, pi=0.75)
            dists['bias_posterior'] = DiagonalNormal(mean=torch.Tensor(self.out_channels).uniform_(-0.01, 0.01),
                                                     rho=torch.Tensor(self.out_channels).normal_(-9, 0.001))

        # specify all distributions that are not given by the user as the default distribution
        for d in dists:
            if getattr(self, d) is None:
                setattr(self, d, dists[d])

        if isinstance(self.weight_prior, PriorNormal) and isinstance(self.weight_posterior, DiagonalNormal):
            self.kl_weights_closed = True
            self.log.debug('Kullback Leibler Divergence for weights will be calculated in closed form.')

        if isinstance(self.bias_prior, PriorNormal) and isinstance(self.bias_posterior, DiagonalNormal):
            self.kl_bias_closed = True
            self.log.debug('Kullback Leibler Divergence for biases will be calculated in closed form.')

    def closed_form_kl(self, bias=False):
        if bias:
            sigma_prior = self.bias_prior.get_std()
            sigma_posterior = self.bias_posterior.get_std()
            mean_prior = self.bias_prior.get_mean()
            mean_posterior = self.bias_posterior.get_mean()
        else:
            sigma_prior = self.weight_prior.get_std()
            sigma_posterior = self.weight_posterior.get_std()
            mean_prior = self.weight_prior.get_mean()
            mean_posterior = self.weight_posterior.get_mean()

        return (torch.log(sigma_prior / sigma_posterior) + (
                    sigma_posterior ** 2 + (mean_posterior - mean_prior) ** 2) / (2 * sigma_prior ** 2) - 0.5).sum()


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

        if self.kl_weights_closed:
            kl = self.closed_form_kl()
        else:
            kl = self.weight_posterior.log_prob(weights).sum() - self.weight_prior.log_prob(weights).sum()
        if self.bias:
            if self.kl_bias_closed:
                kl = kl + self.closed_form_kl(bias=True)
            else:
                kl = kl + self.bias_posterior.log_prob(bias).sum() - self.bias_prior.log_prob(bias).sum()

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

        if self.kl_weights_closed:
            kl = self.closed_form_kl()
        else:
            kl = self.weight_posterior.log_prob(weights).sum() - self.weight_prior.log_prob(weights).sum()
        if self.bias:
            if self.kl_bias_closed:
                kl = kl + self.closed_form_kl(bias=True)
            else:
                kl = kl + self.bias_posterior.log_prob(bias).sum() - self.bias_prior.log_prob(bias).sum()

        return out, kl

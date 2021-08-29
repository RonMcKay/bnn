# Standard Library
import logging
import math

# Thirdparty libraries
import torch
import torch.nn as nn
import torch.nn.functional as F


class VariationalDistribution(nn.Module):
    def __init__(self):
        super().__init__()

    def sample(self):
        raise NotImplementedError('Sampling method has to be implemented!')

    def pdf(self, sample):
        raise NotImplementedError('Probability density function has to be implemented!')

    def log_prob(self, sample):
        raise NotImplementedError('Logarithmic probability density function has to be implemented!')


class DiagonalNormal(VariationalDistribution):
    def __init__(self, mean=torch.tensor(0.0), rho=torch.tensor(0.0), static=False):
        super().__init__()
        self.log = logging.getLogger(__name__[:__name__.rfind('.')] + '.' + type(self).__name__)
        self.mean = mean
        self.rho = rho
        self.last_mean = None
        self.last_rho = None
        if not static:
            self.mean = nn.Parameter(mean)
            self.rho = nn.Parameter(rho)

    def sample(self, **kwargs):
        mean, rho = self.get_params(**kwargs)
        std_dev = self.get_std()
        return mean + std_dev * torch.randn_like(rho)

    def pdf(self, sample, **kwargs):
        mean, rho = self.get_params(**kwargs)
        if sample.size() != mean.size():
            raise ValueError('sample does not match with the distribution shape')
        std_dev = self.get_std()
        return sample.sub(mean).div(std_dev).pow(2).div(-2.0).exp().div((2 * math.pi).sqrt() * std_dev)

    def log_prob(self, sample, **kwargs):
        mean, rho = self.get_params(**kwargs)
        if sample.size() != mean.size():
            raise ValueError('sample does not match with the distribution shape')
        std_dev = self.get_std()
        return torch.log(torch.tensor(2.0 * math.pi)).div(-2.0) - std_dev.log() - sample.sub(mean).div(
            std_dev).pow(2).div(2.0)

    def get_std(self):
        return F.softplus(self.last_rho if self.last_rho is not None else self.rho) + 1e-6

    def get_mean(self, mean=None):
        return self.last_mean if self.last_mean is not None else self.mean

    def get_params(self, **kwargs):
        if 'mean' in kwargs and 'rho' in kwargs:
            self.last_mean = kwargs['mean']
            self.last_rho = kwargs['rho']
            return kwargs['mean'], kwargs['rho']
        else:
            return self.mean, self.rho


class Uniform(VariationalDistribution):
    def __init__(self, lower_bound=torch.tensor(0.0), upper_bound=torch.tensor(1.0), static=False):
        super().__init__()
        self.log = logging.getLogger(__name__[:__name__.rfind('.')] + '.' + type(self).__name__)
        self.log = logging.getLogger(__name__)
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.last_lower_bound = None
        self.last_upper_bound = None
        if not static:
            self.lower_bound = nn.Parameter(self.lower_bound)
            self.upper_bound = nn.Parameter(self.upper_bound)

    def sample(self, **kwargs):
        lower_bound, upper_bound = self.get_params(**kwargs)
        return torch.rand_like(lower_bound).mul(upper_bound - lower_bound).add(lower_bound)

    def pdf(self, sample, **kwargs):
        lower_bound, upper_bound = self.get_params(**kwargs)
        if sample.size() != lower_bound.size():
            raise ValueError('sample does not match with the distribution shape')

        under_lower_bound = sample.lt(lower_bound)
        over_upper_bound = sample.gt(upper_bound)
        pdf = upper_bound.sub(lower_bound).reciprocal()
        pdf[under_lower_bound] = torch.tensor(0.0)
        pdf[over_upper_bound] = torch.tensor(0.0)
        return pdf

    def log_prob(self, sample, **kwargs):
        lower_bound, upper_bound = self.get_params(**kwargs)
        if sample.size() != lower_bound.size():
            raise ValueError('sample does not match with the distribution shape')

        under_lower_bound = sample.lt(lower_bound)
        over_upper_bound = sample.gt(upper_bound)
        mask = (under_lower_bound + over_upper_bound) > 0
        pdf = self.pdf(sample)
        log_prob = torch.zeros_like(pdf)
        log_prob[1 - mask] = pdf[1 - mask].log()
        if mask.sum() > 0:
            self.log.error('Encountered values outside of distribution. '
                           + 'Can not compute log_prob.')
            raise ValueError('Encountered values outside of distribution. '
                             + 'Can not compute log_prob.')
        return log_prob

    def get_params(self, **kwargs):
        if 'lower_bound' in kwargs and 'upper_bound' in kwargs:
            self.last_lower_bound = kwargs['lower_bound']
            self.last_upper_bound = kwargs['upper_bound']
            return kwargs['lower_bound'], kwargs['upper_bound']
        else:
            return self.lower_bound, self.upper_bound


class Bernoulli(VariationalDistribution):
    # Not finished yet
    def __init__(self, probs=torch.tensor(0.5)):
        super().__init__()
        self.log = logging.getLogger(__name__[:__name__.rfind('.')] + '.' + type(self).__name__)
        self.probs = nn.Parameter(probs)


class Beta(VariationalDistribution):
    # Not finished yet
    def __init__(self, concentration1=torch.tensor(0.5),
                 concentration2=torch.tensor(0.5)):
        super().__init__()
        self.log = logging.getLogger(__name__[:__name__.rfind('.')] + '.' + type(self).__name__)
        self.concentration1 = nn.Parameter(concentration1)
        self.concentration2 = nn.Parameter(concentration2)

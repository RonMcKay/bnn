import math

import torch
import torch.nn as nn


class PriorDistribution(nn.Module):
    def __init__(self):
        super().__init__()

    def log_prob(self, sample):
        raise NotImplementedError(
            "Logarithmic probability density function has to be implemented!"
        )


class DiagonalNormal(PriorDistribution):
    def __init__(self, mean=0, std=1):
        super().__init__()
        if not isinstance(mean, torch.FloatTensor):
            mean = torch.tensor(mean, dtype=torch.float)
        if not isinstance(std, torch.FloatTensor):
            std = torch.tensor(std, dtype=torch.float)
        self.register_buffer("mean", mean.clone())
        self.register_buffer("std", std.clone())

    def log_prob(self, sample):
        return (
            torch.tensor(2.0 * math.pi).log() * (-0.5)
            - self.std.log()
            + ((sample - self.mean) / self.std) ** 2 * (-0.5)
        )

    def get_std(self):
        return self.std

    def get_mean(self):
        return self.mean

    def extra_repr(self):
        return "mean={}, std={}".format(
            round(self.mean.item(), 5), round(self.std.item(), 5)
        )


class Laplace(PriorDistribution):
    def __init__(self, mean, std):
        super().__init__()
        if not isinstance(mean, torch.FloatTensor):
            mean = torch.tensor(mean, dtype=torch.float)
        if not isinstance(std, torch.FloatTensor):
            std = torch.tensor(std, dtype=torch.float)
        self.register_buffer("mean", mean.clone())
        self.register_buffer("std", std.clone())

    def log_prob(self, sample):
        return sample.sub(self.mean).abs().div(self.std).neg() - self.std.mul(2).log()


class GaussianMixture(PriorDistribution):
    """Scale mixture of two Gaussian densities
    from 'Weight Uncertainty in Neural Networks'
    """

    def __init__(self, sigma1, sigma2, pi, mu1=0, mu2=0):
        super().__init__()
        if not isinstance(pi, torch.FloatTensor):
            pi = torch.tensor(pi, dtype=torch.float)
        self.register_buffer("pi", pi.clone())

        self.normal1 = DiagonalNormal(mean=mu1, std=sigma1)
        self.normal2 = DiagonalNormal(mean=mu2, std=sigma2)

    def log_prob(self, value):
        logprob1 = self.normal1.log_prob(value)
        logprob2 = self.normal2.log_prob(value)

        # Numerical stability trick -> unnormalising logprobs will underflow otherwise
        # from: https://github.com/JavierAntoran/Bayesian-Neural-Networks
        max_logprob = torch.max(logprob1, logprob2)
        # normalised_probs = self.pi * torch.exp(logprob1 - max_logprob) \
        #     + (1-self.pi) * torch.exp(logprob2 - max_logprob)
        # logprob = torch.log(normalised_probs) + max_logprob
        # return logprob
        return (
            torch.log(
                self.pi * torch.exp(logprob1 - max_logprob)
                + (1 - self.pi) * torch.exp(logprob2 - max_logprob)
            )
            + max_logprob
        )

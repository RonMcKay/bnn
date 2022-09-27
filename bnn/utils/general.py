from collections import OrderedDict
from itertools import islice
import logging
import operator

from bnn import BayesianLayer
from bnn.distributions.posteriors import DiagonalNormal, VariationalDistribution
from bnn.distributions.priors import DiagonalNormal as PriorNormal
from bnn.distributions.priors import PriorDistribution
import torch
import torch.nn as nn


class Sequential(BayesianLayer):
    def __init__(self, *args):
        super().__init__()
        if len(args) == 1 and isinstance(args[0], OrderedDict):
            for key, module in args[0].items():
                self.add_module(key, module)
        else:
            for idx, module in enumerate(args):
                self.add_module(str(idx), module)

    def _get_item_by_idx(self, iterator, idx):
        """Get the idx-th item of the iterator"""
        size = len(self)
        idx = operator.index(idx)
        if not -size <= idx < size:
            raise IndexError("index {} is out of range".format(idx))
        idx %= size
        return next(islice(iterator, idx, None))

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return self.__class__(OrderedDict(list(self._modules.items())[idx]))
        else:
            return self._get_item_by_idx(self._modules.values(), idx)

    def __setitem__(self, idx, module):
        key = self._get_item_by_idx(self._modules.keys(), idx)
        return setattr(self, key, module)

    def __delitem__(self, idx):
        if isinstance(idx, slice):
            for key in list(self._modules.keys())[idx]:
                delattr(self, key)
        else:
            key = self._get_item_by_idx(self._modules.keys(), idx)
            delattr(self, key)

    def __len__(self):
        return len(self._modules)

    def __dir__(self):
        keys = super().__dir__()
        keys = [key for key in keys if not key.isdigit()]
        return keys

    def forward(self, x):
        kl = torch.tensor(0.0, device=x.device, requires_grad=True)
        for module in self._modules.values():
            if isinstance(module, BayesianLayer):
                x, _kl = module(x)
                kl = kl + _kl
            else:
                x = module(x)
        return x, kl


class KLLoss(nn.Module):
    def __init__(
        self,
        likelihood_cost=nn.CrossEntropyLoss(
            ignore_index=-100, weight=None, reduction="sum"
        ),
    ):
        super().__init__()
        self.log = logging.getLogger(__name__ + ".KLLoss")
        self.likelihood_cost = likelihood_cost
        self.log.debug("Initialized Kullback-Leibler loss")

    def forward(self, outputs, target, kl, batch_weight, **kwargs):
        # for ways of setting batch_weight see e.g. 'get_beta' in
        # https://github.com/kumar-shridhar/PyTorch-BayesianCNN/blob/master/metrics.py

        loss = []
        # first dimension is number of samples drawn
        # this is why we have to iterate over it to apply the
        # likelihood cost to each sample
        for i in range(outputs.size(0)):
            loss.append(self.likelihood_cost(outputs[i], target))
        loss = torch.stack(loss).mean()
        kl_loss = batch_weight * kl
        if "_run" in kwargs:
            # automatic logging for sacred users if _run instance is supplied
            kwargs["_run"].log_scalar("train.cross_entropy", loss.item())
            kwargs["_run"].log_scalar("train.kl_loss", kl_loss.item())
        self.log.debug(
            "Cross Entropy: {:.2f}, KL: {:.2f}".format(loss.item(), kl_loss.item())
        )

        return kl_loss + loss


def kldivergence(
    prior: PriorDistribution, posterior: VariationalDistribution, sample: torch.Tensor
):
    if isinstance(prior, PriorNormal) and isinstance(posterior, DiagonalNormal):
        # calculate Kullback-Leibler Divergence in closed form
        sigma_prior = prior.get_std()
        sigma_posterior = posterior.get_std()
        mean_prior = prior.get_mean()
        mean_posterior = posterior.get_mean()
        return (
            torch.log(sigma_prior / sigma_posterior)
            + (sigma_posterior**2 + (mean_posterior - mean_prior) ** 2)
            / (2 * sigma_prior**2)
            - 0.5
        ).sum()
    else:
        return posterior.log_prob(sample).sum() - prior.log_prob(sample).sum()

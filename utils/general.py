import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import logging
import operator

class BayesianLayer(nn.Module):
    def __init__(self):
        super().__init__()
        
class Sequential(nn.Module):
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
            raise IndexError('index {} is out of range'.format(idx))
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
    
    def _delitem__(self, idx):
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
                
    def forward(self, input):
        kl = torch.tensor(0.0, device=input.device, requires_grad=True)
        for module in self._modules.values():
            if isinstance(module, BayesianLayer):
                input, _kl = module(input)
                kl = kl + _kl
            else:
                input = module(input)
        return input, kl
        
class KLLoss(nn.Module):
    def __init__(self, ignore_index=-100, weight=None, reduction='sum'):
        super().__init__()
        self.log = logging.getLogger(__name__ + '.KLLoss')
        self.likelihood_cost = nn.CrossEntropyLoss(reduction=reduction,
                                                   ignore_index=ignore_index,
                                                   weight=weight)
        self.log.debug('Initialized Kullback-Leibler loss')
        
    def forward(self, outputs, target, kl, batch_weight, **kwargs):
        loss = []
        for i in range(outputs.size(0)):
            loss.append(self.likelihood_cost(outputs[i], target))
        loss = torch.stack(loss).mean()
        kl_loss = batch_weight * kl
        if '_run' in kwargs:
            kwargs['_run'].log_scalar('train.cross_entropy', loss.item())
            kwargs['_run'].log_scalar('train.kl_loss', kl_loss.item())
        self.log.debug('Cross Entropy: {:.2f}, KL: {:.2f}'.format(loss.item(), kl_loss.item()))
        
        return kl_loss + loss

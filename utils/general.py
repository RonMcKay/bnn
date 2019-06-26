import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import logging

class BayesianLayer(nn.Module):
    def __init__(self):
        super().__init__()
        
class KLLoss(nn.Module):
    def __init__(self, ignore_index=-100):
        super().__init__()
        self.log = logging.getLogger(__name__ + '.KLLoss')
        self.likelihood_cost = nn.CrossEntropyLoss(reduction='sum', ignore_index=ignore_index)
        self.log.debug('Initialized Kullback-Leibler loss')
        
    def forward(self, outputs, target, kl, batch_weight):
        loss = []
        for i in range(outputs.size(0)):
            loss.append(self.likelihood_cost(outputs[i], target))
        loss = torch.stack(loss).mean()
        kl_loss = batch_weight * kl
        self.log.debug('Cross Entropy: {:.2f}, KL: {:.2f}'.format(loss.item(), kl_loss.item()))
        
        return kl_loss + loss